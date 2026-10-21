// The existing MediaPipe CPU model, moved off the UI thread. One in-flight
// frame only: no inference backlog and no second webcam stream.
self.exports={};
importScripts('/static/deps/mediapipe-vision-0.10.21.js');
const {FilesetResolver,FaceLandmarker}=self.exports;
let model;
onmessage=async({data:d})=>{try{
 if(d.type==='load'){
  const files=await FilesetResolver.forVisionTasks('/static/deps/wasm');
  model=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/static/deps/face_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numFaces:4,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
  postMessage({type:'ready'});return;
 }
 const start=performance.now();let result;
 try{result=model.detectForVideo(d.frame,d.at);}finally{d.frame.close();}
 postMessage({type:'result',result,at:d.at,ms:performance.now()-start,generation:d.generation});
}catch(e){d.frame?.close();postMessage({type:'error',error:e.message});}};
