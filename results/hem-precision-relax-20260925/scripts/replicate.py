# replicate.py SRC.bin OUT.bin INDEX[,INDEX...] COPIES : batch of the selected cells, each repeated
import struct,sys
src,out,idx,copies=sys.argv[1],sys.argv[2],[int(x) for x in sys.argv[3].split(',')],int(sys.argv[4])
d=open(src,'rb').read();h=list(struct.unpack_from('<8Q',d));mb,n,ns,sb,cb=h[2],h[3],h[4],h[5],h[6]
o=64+mb;q=d[o:o+n*ns*8];o+=n*ns*8;e=d[o:o+n*8];o+=n*8;st=d[o:o+n*sb];o+=n*sb;cp=d[o:o+n*cb]
sel=[i for i in idx for _ in range(copies)];h[3]=len(sel)
with open(out,'wb') as f:
    f.write(struct.pack('<8Q',*h));f.write(d[64:64+mb])
    for arr,w in ((q,ns*8),(e,8),(st,sb),(cp,cb)):
        for i in sel:f.write(arr[i*w:(i+1)*w])
