// SPDX-License-Identifier: GPL-3.0-or-later
// Included after Flow's host Face definition. Only the opt-in Cartesian
// geometry mode calls this mesh inspection; legacy transport does not scan
// points or allocate logical maps.
struct CartesianHostGeometry {
    std::array<uint64_t,3> dims{};
    std::array<double,3> origin{},spacing{};
    std::vector<int64_t> cellFromLogical;
    std::vector<double> cellCenterXYZ,faceCenterXYZ;
};

CartesianHostGeometry makeCartesianHostGeometry(const fvMesh& mesh,
    const std::vector<Face>& transportFaces,const Array& volumes) {
    const auto points=host(mesh.points());
    const auto centres=host(mesh.C().primitiveField());
    const auto faceCentres=host(mesh.faceCentres());
    demand(!points.empty()&&centres.size()==volumes.size()&&!transportFaces.empty(),
        "Empty Cartesian mesh geometry");
    CartesianHostGeometry out{};
    std::array<double,3> high{};
    for(int d=0;d<3;++d) {
        out.origin[d]=std::numeric_limits<double>::infinity();
        high[d]=-std::numeric_limits<double>::infinity();
    }
    for(const auto& p:points)for(int d=0;d<3;++d) {
        demand(std::isfinite(p[d]),"Nonfinite Cartesian mesh point");
        out.origin[d]=std::min(out.origin[d],double(p[d]));
        high[d]=std::max(high[d],double(p[d]));
    }
    for(int d=0;d<3;++d) {
        const double length=high[d]-out.origin[d];
        demand(std::isfinite(length)&&length>0,"Degenerate Cartesian domain");
        std::vector<double> levels;levels.reserve(centres.size());
        for(const auto& c:centres) {
            demand(std::isfinite(c[d]),"Nonfinite Cartesian cell centre");
            levels.push_back(c[d]);
        }
        std::sort(levels.begin(),levels.end());
        const double grouping=1e-10*length;
        size_t distinct=0;double last=0;
        for(double value:levels)if(!distinct||value-last>grouping) {
            last=value;++distinct;
        }
        demand(distinct>=8&&distinct<=size_t(INT64_MAX),
            "Cartesian mode requires at least eight uniform cells per axis");
        out.dims[d]=distinct;out.spacing[d]=length/double(distinct);
        demand(std::isfinite(out.spacing[d])&&out.spacing[d]>0,
            "Invalid Cartesian cell spacing");
    }
    demand(out.dims[0]<=std::numeric_limits<size_t>::max()/out.dims[1]
        &&out.dims[0]*out.dims[1]<=std::numeric_limits<size_t>::max()/out.dims[2]
        &&out.dims[0]*out.dims[1]*out.dims[2]==centres.size(),
        "Cartesian centre levels do not fill a tensor-product mesh");
    const size_t nc=centres.size();
    out.cellFromLogical.assign(nc,-1);out.cellCenterXYZ.resize(3*nc);
    std::vector<uint64_t> logicalFromCell(nc);
    const double volume=out.spacing[0]*out.spacing[1]*out.spacing[2];
    for(size_t c=0;c<nc;++c) {
        std::array<uint64_t,3> ijk{};
        demand(std::isfinite(volumes[c])&&std::abs(volumes[c]-volume)<=1e-8*volume,
            "Cartesian cell volume differs from its rectangular bounds");
        for(int d=0;d<3;++d) {
            const double coordinate=centres[c][d];
            const double position=(coordinate-out.origin[d])/out.spacing[d]-.5;
            demand(std::isfinite(position)&&position>=-1e-8
                &&position<=double(out.dims[d]-1)+1e-8,
                "Cartesian cell centre is outside the inferred grid");
            const auto index=std::llround(position);
            demand(index>=0&&uint64_t(index)<out.dims[d]
                &&std::abs(position-double(index))<=1e-8,
                "Cartesian cell centre is off a uniform logical index");
            ijk[d]=uint64_t(index);out.cellCenterXYZ[3*c+d]=coordinate;
        }
        const size_t logical=ijk[0]+out.dims[0]*(ijk[1]+out.dims[1]*ijk[2]);
        demand(out.cellFromLogical[logical]<0,"Duplicate Cartesian logical cell");
        out.cellFromLogical[logical]=int64_t(c);
        logicalFromCell[c]=logical;
    }
    for(int64_t cell:out.cellFromLogical)
        demand(cell>=0,"Missing Cartesian logical cell");
    out.faceCenterXYZ.resize(3*transportFaces.size());
    const auto& polygons=mesh.faces();
    auto corners=[&](label meshFace,int axis,int sign,
                     const std::array<uint64_t,3>& ijk,bool peer) {
        demand(meshFace>=0&&size_t(meshFace)<polygons.size(),"Invalid Cartesian polygon index");
        const auto& polygon=polygons[meshFace];
        demand(polygon.size()==4,"Cartesian face is not a quadrilateral");
        int tangential[2]{},next=0;
        for(int d=0;d<3;++d)if(d!=axis)tangential[next++]=d;
        const double plane=out.origin[axis]+
            (peer?(sign>0?0.:double(out.dims[axis])):
                (double(ijk[axis])+(sign>0?1.:0.)))*out.spacing[axis];
        unsigned mask=0;
        for(label p:polygon) {
            demand(p>=0&&size_t(p)<points.size(),"Cartesian face has invalid point index");
            const auto& vertex=points[p];
            demand(std::abs(double(vertex[axis])-plane)<=1e-8*out.spacing[axis],
                "Cartesian face point is off its normal plane");
            unsigned bit=0;
            for(int t=0;t<2;++t) {
                const int d=tangential[t];
                const double lower=out.origin[d]+double(ijk[d])*out.spacing[d];
                const double upper=lower+out.spacing[d];
                const double value=vertex[d];
                if(std::abs(value-upper)<=1e-8*out.spacing[d])bit|=1u<<t;
                else demand(std::abs(value-lower)<=1e-8*out.spacing[d],
                    "Cartesian face point is off its rectangular corner");
            }
            demand(!(mask&(1u<<bit)),"Duplicate Cartesian face corner");
            mask|=1u<<bit;
        }
        demand(mask==15,"Cartesian face does not cover all rectangle corners");
    };
    for(size_t fi=0;fi<transportFaces.size();++fi) {
        const auto& f=transportFaces[fi];
        demand(f.owner>=0&&size_t(f.owner)<nc,"Invalid Cartesian owner cell");
        int axis=0;for(int d=1;d<3;++d)
            if(std::abs(f.normal[d])>std::abs(f.normal[axis]))axis=d;
        const int sign=f.normal[axis]>=0?1:-1;
        const uint64_t logical=logicalFromCell[size_t(f.owner)];
        const std::array<uint64_t,3> ijk{logical%out.dims[0],
            (logical/out.dims[0])%out.dims[1],logical/(out.dims[0]*out.dims[1])};
        corners(f.meshFace,axis,sign,ijk,false);
        if(f.meshPeerFace>=0)corners(f.meshPeerFace,axis,sign,ijk,true);
        demand(f.meshFace>=0&&size_t(f.meshFace)<faceCentres.size(),
            "Invalid Cartesian face centre index");
        for(int d=0;d<3;++d)out.faceCenterXYZ[3*fi+d]=faceCentres[f.meshFace][d];
    }
    return out;
}
