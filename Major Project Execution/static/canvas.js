

window.addEventListener('load',()=>{
    console.log("hello")

    const canvas = document.querySelector("#canvas");

    const ctx = canvas.getContext("2d");
    const canvasoffsetX = canvas.offsetLeft
    const canvasoffsetY = canvas.offsetTop
    console.log(canvasoffsetX)
    console.log(canvasoffsetY)

    canvas.height  = 400 - canvasoffsetY ;
    
    canvas.width = 700 - canvasoffsetX;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width,canvas.height )




    let painting  = false;
    
    function startPosition()
    {
        painting = true;
    }

    function finishedPosition()
    {
        painting = false;
        ctx.beginPath()

    }

    function draw(e)
    {
        if (!painting) return;

        ctx.lineWidth = 4;
        ctx.lineCap = "square";


        ctx.lineTo(e.clientX-canvasoffsetX,e.clientY-canvasoffsetY);
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(e.clientX-canvasoffsetX,e.clientY-canvasoffsetY)

    }


    canvas.addEventListener("mousedown",startPosition);
    canvas.addEventListener("mouseup",finishedPosition);
    canvas.addEventListener("mousemove",draw)



});


function init() {
    console.log("inside init")
    document.getElementById('image').addEventListener('change', function() {

        var reader = new FileReader();
        var fileToRead = document.getElementById('image').files[0];
        // var GetFile = new FileReader();
      
        reader.addEventListener("loadend", function() {
            // reader.result contains the contents of blob as a typed array
            // we insert content of file in DOM here
            console.log(reader.result)
            console.log(reader.result.length)
            document.getElementById("imgsrc").value = reader.result;
         });
          
        //   GetFile.readAsText(this.files[0]);
          reader.readAsDataURL(fileToRead)
          console.log(reader.result)
          console.log(typeof(document.getElementById('image').files[0]))
          console.log(typeof(document.getElementById('image').files[0]))
        //   GetFile.readAsText(document.getElementById('image').files[0])
      })
  }

function getImage(id)
{
    if(id=="canvaimage")
    {
img = document.getElementById("canvas")
    
dataURL = img.toDataURL()
console.log(dataURL)
console.log(typeof(dataURL))

// form.file=dataUrl
// console.log(form.file)
localStorage.setItem("imgData", dataURL);
document.getElementById("imgsrc").value=dataURL;
    }
    else if (id=="image")
    {

    // var file = document.getElementById('image');

    // if(file.files.length)
    // {
    //     var reader = new FileReader();

    //     reader.onload = function(e)
    //     {
    //         document.getElementById('imgsrc').innerHTML = e.target.result;
    //         console.log(e.target.result)
    //     };

    //     reader.readAsBinaryString(file.files[0]);
    //     console.log(e.target.result)
        
    // }
    }
// fs.writeFile( "E:\\Major Project Execution\\static", "b", dataURL)

// fetch("http://localhost:5000/input", {
// method: "POST",
// mode: 'cors',
// body: dataURL,
// headers: {
// "Content-type": "application/json; charset=UTF-8"
// }
// })
// .then((response) => response)
// .then((json) => console.log(json));

}
;

// function RunFile() {
//     WshShell = new ActiveXObject("WScript.Shell");
//     WshShell.Run("C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Accessories\\Paint.lnk", 1, false);
//     }

function convert()
{
    console.log("inside convert")
    
    

    let txt = document.getElementById("equations").innerHTML;
    console.log(typeof(txt))
    console.log(txt)

    for (var i = 0;i<txt.length;i++)
    {
    if (txt.includes("**"))
    {
        let pos=txt.indexOf("**")
        let l = txt.length;
        txt=txt.slice(0,pos)+txt[pos+2].sup()+txt.slice(pos+3,l)
    }
    else if(txt.includes("[[")){
        let pos=txt.indexOf("[[")
        let l = txt.length;
        txt=txt.slice(0,pos)+txt[pos+2].sub()+txt.slice(pos+3,l)

    }
}
    console.log(txt)

    document.getElementById("equations").innerHTML=txt
    // document.getElementById("result").innerHTML = txt
    
    
}


function savepdf()
{
    equations = document.getElementById("equations").innerHTML
    result = document.getElementById("equations").innerHTML
    console.log("saving to pdf")
    console.log(equations)



    var preHtml = "<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'><head><meta charset='utf-8'><title>Export HTML To Doc</title></head><body>";
    var postHtml = "</body></html>";
    var html = preHtml+document.getElementById("equations").innerHTML+postHtml;

    var blob = new Blob(['\ufeff', html], {
        type: 'application/msword'
    });
    
    // Specify link url
    var url = 'data:application/vnd.ms-word;charset=utf-8,' + encodeURIComponent(html);
    
    // Specify file name
    var filename = 'document.doc';
    
    // Create download link element
    var downloadLink = document.createElement("a");

    document.body.appendChild(downloadLink);
    
    if(navigator.msSaveOrOpenBlob ){
        navigator.msSaveOrOpenBlob(blob, filename);
    }else{
        // Create a link to the file
        downloadLink.href = url;
        
        // Setting the file name
        downloadLink.download = filename;
        
        //triggering the function
        downloadLink.click();
    }
    
    document.body.removeChild(downloadLink);





}