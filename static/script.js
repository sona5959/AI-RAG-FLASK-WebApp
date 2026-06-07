document.getElementById("rag-form").onsubmit= async(e) => {
    e.preventDefault();
    let formData = new FormData(e.target);

    let processing = document.getElementById("processing");

    processing.style.display = "block";

    let result = await fetch('/upload', {
            method: "POST",
            body: formData
        }
    );

    let data = await result.text();

    // console.log(data.response);

    document.getElementById("answer").innerHTML =data;
    clean_data();

    processing.style.display = "none"


};
function clean_data(){
  res=document.getElementById("answer")
  clean_res=document.getElementById("clean-rag")
    if(res && clean_res){
      if(res.textContent.trim() !== "")
      {
         clean_res.style.display = "block";
      }
     else
     {
        clean_res.style.display = "none";
     }
   }

}

// document.getElementById("clean-reg").addEventListener("click", clean_data);


