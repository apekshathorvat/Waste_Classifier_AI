let video=document.getElementById("camera");


if(video){

navigator.mediaDevices.getUserMedia({
video:true
})
.then(stream=>{

video.srcObject=stream;

})

}