// WebCrypto is unavailable on ordinary LAN HTTP. Keep asset verification there too.
const primes=[];
for(let n=2;primes.length<64;n++)if(!primes.some(p=>p*p<=n&&n%p===0))primes.push(n);
const fraction=x=>Math.floor((x-Math.floor(x))*0x100000000)>>>0;
const initial=primes.slice(0,8).map(p=>fraction(Math.sqrt(p)));
const constants=primes.map(p=>fraction(Math.cbrt(p)));
const rotate=(x,n)=>(x>>>n)|(x<<(32-n));

export async function sha256(buffer,subtle=globalThis.crypto?.subtle) {
  if(subtle){const hash=await subtle.digest('SHA-256',buffer);return Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join('');}
  const input=new Uint8Array(buffer);
  if(input.length>64*1024*1024)throw new Error('Asset exceeds verification limit');
  const size=Math.ceil((input.length+9)/64)*64,bytes=new Uint8Array(size),view=new DataView(bytes.buffer);
  bytes.set(input);bytes[input.length]=0x80;
  view.setUint32(size-8,Math.floor(input.length/0x20000000));view.setUint32(size-4,(input.length*8)>>>0);
  const h=initial.slice(),w=new Uint32Array(64);
  for(let offset=0;offset<size;offset+=64){
    for(let i=0;i<16;i++)w[i]=view.getUint32(offset+i*4);
    for(let i=16;i<64;i++){const x=w[i-15],y=w[i-2];w[i]=((rotate(x,7)^rotate(x,18)^(x>>>3))+w[i-16]+(rotate(y,17)^rotate(y,19)^(y>>>10))+w[i-7])>>>0;}
    let [a,b,c,d,e,f,g,k]=h;
    for(let i=0;i<64;i++){
      const t1=(k+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+constants[i]+w[i])>>>0;
      const t2=((rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c)))>>>0;
      k=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;
    }
    [a,b,c,d,e,f,g,k].forEach((v,i)=>h[i]=(h[i]+v)>>>0);
  }
  return h.map(v=>v.toString(16).padStart(8,'0')).join('');
}
