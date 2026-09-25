# concat.py OUT A.bin[:count] B.bin[:count] ... : one batch from the first `count` cells of each capture
import struct,sys
def load(p):
    path,_,k=p.partition(':');d=open(path,'rb').read();h=list(struct.unpack_from('<8Q',d));mb,n,ns,sb,cb=h[2],h[3],h[4],h[5],h[6]
    k=int(k) if k else n;o=64+mb;parts=[]
    for w in (ns*8,8,sb,cb): parts.append(d[o:o+k*w]);o+=n*w
    return h,d[64:64+mb],k,parts
xs=[load(p) for p in sys.argv[2:]];h=xs[0][0][:];h[3]=sum(x[2] for x in xs)
with open(sys.argv[1],'wb') as f:
    f.write(struct.pack('<8Q',*h));f.write(xs[0][1])
    for i in range(4):
        for x in xs: f.write(x[3][i])
