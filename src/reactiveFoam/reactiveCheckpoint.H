// SPDX-License-Identifier: GPL-3.0-or-later
// Schema 3: native FP64, exact conserved data and seeds, durable directory commit.
struct CheckpointHistory {
    Array initial,boundary;
    uint64_t steps=0,retries=0;
    double time=0,lastDt=0,velocityScale=1,energyScale=0;
    bool loaded=false;
};
namespace checkpoint {
namespace fs=std::filesystem;
inline void requireIO(bool ok,const std::string& message) {
    if(!ok)throw PersistentIOError("Failed to write checkpoint "+message);
}
inline std::string sha(const fs::path& path) {
    char hash[65];
    if(pintle_rt_file_sha256_v1(path.c_str(),hash,sizeof(hash)))
        throw PersistentIOError("Cannot hash checkpoint file "+path.string());
    return hash;
}
inline void sync(const fs::path& path,bool directory=false) {
    const int fd=::open(path.c_str(),O_RDONLY|(directory?O_DIRECTORY:0));
    requireIO(fd>=0,"open for fsync: "+path.string());
    const int status=::fsync(fd);::close(fd);requireIO(status==0,"fsync: "+path.string());
}
template<class T> void put(std::ostream& out,const T& value) {
    static_assert(std::is_trivially_copyable<T>::value,"binary POD required");
    out.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
template<class T> void get(std::istream& in,T& value) {
    static_assert(std::is_trivially_copyable<T>::value,"binary POD required");
    in.read(reinterpret_cast<char*>(&value),sizeof(value));
    demand(bool(in),"Incomplete checkpoint state");
}
void save(const fs::path& path,const Array& q,const States& states,bool mechanical,const CheckpointHistory& h) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);requireIO(bool(out),path.string());
    const uint64_t magic=0x524643484b505433ULL,endian=0x0102030405060708ULL;
    put(out,magic);put(out,endian);put(out,uint64_t(sizeof(double)));put(out,uint64_t(states.size()));
    put(out,uint64_t(h.initial.size()));put(out,uint64_t(mechanical?sizeof(CellState):sizeof(PintleThermoState)));
    put(out,h.steps);put(out,h.retries);put(out,h.time);put(out,h.lastDt);put(out,h.velocityScale);put(out,h.energyScale);
    for(double value:h.initial)put(out,value);for(double value:h.boundary)put(out,value);
    out.write(reinterpret_cast<const char*>(q.data()),q.size()*sizeof(double));
    demand(states.isMechanical()==mechanical,"Checkpoint active state mode mismatch");
    for(size_t c=0;c<states.size();++c) {
        if(mechanical)put(out,states.full(c));else put(out,states[c]);
    }
    out.flush();requireIO(bool(out),path.string());out.close();requireIO(!out.fail(),path.string());
}
void verify(const fs::path& dir) {
    std::ifstream marker(dir/"reactiveCheckpointComplete");
    std::string signature;std::getline(marker,signature);
    demand(signature=="ReactiveFoam checkpoint 3","Incomplete checkpoint: missing/invalid complete marker");
    std::string hash,name;bool state=false,identity=false;size_t count=0;
    while(marker>>hash>>std::quoted(name)) {
        demand(pintleIdentityHash(hash)&&name.find('/')==std::string::npos&&name!="."&&name!="..",
               "Invalid checkpoint manifest entry");
        demand(fs::is_regular_file(dir/name)&&sha(dir/name)==hash,"Checkpoint checksum mismatch: "+name);
        state|=name=="reactiveState.bin";
        identity|=name=="reactiveStateIdentity"||name=="reactiveStateIdentity.gz";++count;
    }
    demand(marker.eof()&&state&&identity&&count>2,"Incomplete checkpoint manifest");
}
void load(const fs::path& dir,Array& q,States& states,bool mechanical,size_t nv,CheckpointHistory& h) {
    verify(dir);std::ifstream in(dir/"reactiveState.bin",std::ios::binary);
    uint64_t magic,endian,scalarBytes,cells,variables,stateBytes;
    get(in,magic);get(in,endian);get(in,scalarBytes);get(in,cells);get(in,variables);get(in,stateBytes);
    demand(magic==0x524643484b505433ULL&&endian==0x0102030405060708ULL&&scalarBytes==sizeof(double)
        &&cells==states.size()&&variables==nv&&stateBytes==(mechanical?sizeof(CellState):sizeof(PintleThermoState)),
        "Checkpoint layout/mesh/closure mismatch; explicit migration required");
    get(in,h.steps);get(in,h.retries);get(in,h.time);get(in,h.lastDt);get(in,h.velocityScale);get(in,h.energyScale);
    demand(std::isfinite(h.time)&&std::string(Foam::Time::timeName(h.time).c_str())==dir.filename().string()&&std::isfinite(h.lastDt)&&h.lastDt>=0&&std::isfinite(h.velocityScale)&&h.velocityScale>0
        &&std::isfinite(h.energyScale)&&h.energyScale>0,"Invalid checkpoint time/history");
    h.initial.resize(nv);h.boundary.resize(nv);
    for(double& v:h.initial){get(in,v);demand(std::isfinite(v),"Invalid checkpoint conservation baseline");}
    for(double& v:h.boundary){get(in,v);demand(std::isfinite(v),"Invalid checkpoint boundary integral");}
    for(double& v:q) {
        double exact;get(in,exact);demand(std::isfinite(exact)&&v==exact,"Checkpoint field/binary conserved state mismatch");v=exact;
    }
    demand(states.isMechanical()==mechanical,"Checkpoint active state mode mismatch");
    for(size_t c=0;c<states.size();++c) {
        CellState saved{};if(mechanical)get(in,saved);else get(in,static_cast<PintleThermoState&>(saved));
        demand(states[c].p==saved.p&&states[c].T==saved.T,"Checkpoint field/binary thermodynamic seed mismatch");
        if(mechanical)states.full(c)=saved;else states[c]=saved;
    }
    demand(in.peek()==std::char_traits<char>::eof(),"Checkpoint state has unexpected trailing data");h.loaded=true;
}
class Transaction {
    fs::path root,target,stage;
public:
    Transaction(const std::string& rootPath,const std::string& time,const std::string& id):root(rootPath),target(root/time) {
        static uint64_t sequence=0;stage=root/(".reactive-checkpoint-"+id+"-"+std::to_string(++sequence));
        requireIO(fs::create_directory(stage),"create staging directory: "+stage.string());
        // Retain auxiliary initial fields. Writes only touch the hidden copy.
        try {if(fs::exists(target))for(const auto& entry:fs::directory_iterator(target))
            fs::copy(entry.path(),stage/entry.path().filename(),fs::copy_options::recursive|fs::copy_options::copy_symlinks);
        }catch(...){fs::remove_all(stage);throw;}
        fs::remove(stage/"reactiveCheckpointComplete");
    }
    ~Transaction(){std::error_code ec;fs::remove_all(stage,ec);}
    std::string instance() const {return stage.filename().string();}
    fs::path path() const {return stage;}
    void checkField(const std::string& name) const {
        requireIO(!fs::exists(target/name)||fs::is_regular_file(target/name),name+" is not a regular file");
        // A linked input may refer outside the transaction. Replace the link,
        // never write through it into the previous checkpoint or shared input.
        if(fs::is_symlink(stage/name))fs::remove(stage/name);
    }
    void commit() {
        std::vector<fs::path> files;
        for(const auto& entry:fs::directory_iterator(stage))if(entry.is_regular_file())files.push_back(entry.path());
        std::sort(files.begin(),files.end());
        std::ofstream marker(stage/"reactiveCheckpointComplete",std::ios::trunc);
        marker<<"ReactiveFoam checkpoint 3\n";
        for(const auto& file:files){sync(file);marker<<sha(file)<<' '<<std::quoted(file.filename().string())<<'\n';}
        marker.flush();requireIO(bool(marker),"complete marker");marker.close();requireIO(!marker.fail(),"complete marker close");
        sync(stage/"reactiveCheckpointComplete");sync(stage,true);
        if(fs::exists(target)) {
            requireIO(::syscall(SYS_renameat2,AT_FDCWD,stage.c_str(),AT_FDCWD,target.c_str(),RENAME_EXCHANGE)==0,
                      "atomic directory exchange: "+target.string());
        } else requireIO(::rename(stage.c_str(),target.c_str())==0,"atomic directory rename: "+target.string());
        sync(root,true);
    }
};
}
