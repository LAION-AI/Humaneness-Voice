// Lazy, CPU-only face-api.js recognition net (128 D). MediaPipe supplies the
// aligned crop, so no second detector/landmark/age/gender/emotion model is loaded.
let loaded;
async function load(){
 if(!loaded)loaded=(async()=>{
  importScripts('/static/deps/face-api-0.22.2.js');
  // face-api's environment detector predates dedicated workers. Supply real
  // worker APIs explicitly; inference itself consumes a Tensor3D, not a DOM.
  faceapi.env.setEnv({Canvas:OffscreenCanvas,CanvasRenderingContext2D:OffscreenCanvasRenderingContext2D,
   Image:ImageBitmap,ImageData,Video:class WorkerVideo{},fetch:self.fetch.bind(self),
   createCanvasElement:()=>new OffscreenCanvas(150,150),
   createImageElement:()=>{throw Error('DOM images are not used by this worker');},
   readFile:()=>{throw Error('No filesystem access');}});
  await faceapi.tf.setBackend('cpu');
  await faceapi.nets.faceRecognitionNet.loadFromUri('/static/deps/face-api');
 })();return loaded;
}
onmessage=async({data:d})=>{try{
 await load();const start=performance.now();
 const tensor=faceapi.tf.tensor3d(new Uint8Array(d.pixels),[150,150,3]);
 let face;try{face=Array.from(await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(tensor));}finally{tensor.dispose();}
 postMessage({id:d.id,generation:d.generation,identityEpoch:d.identityEpoch,face,ms:performance.now()-start,backend:faceapi.tf.getBackend()});
}catch(e){loaded=null;postMessage({id:d.id,generation:d.generation,identityEpoch:d.identityEpoch,error:String(e.message)});}};
