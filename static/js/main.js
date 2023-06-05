// Init
document.addEventListener("DOMContentLoaded", function() {
  document.querySelector(".image-section").style.display = "none";
  document.querySelector(".loader").style.display = "none";
  document.querySelector("#result").style.display = "none";

  // Upload Preview
  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function(e) {
        document.querySelector("#imagePreview").style.backgroundImage = "url(" + e.target.result + ")";
        document.querySelector("#imagePreview").style.display = "none";
        document.querySelector("#imagePreview").style.display = "block";
      };
      reader.readAsDataURL(input.files[0]);
    }
  }
  document.querySelector("#imageUpload").addEventListener("change", function() {
    document.querySelector(".image-section").style.display = "block";
    document.querySelector("#btn-predict").style.display = "block";
    document.querySelector("#result").textContent = "";
    document.querySelector("#result").style.display = "none";
    readURL(this);
  });

  // Predict
  document.querySelector("#btn-predict").addEventListener("click", function() {
    var form_data = new FormData(document.querySelector("#upload-file"));

    // Show loading animation
    this.style.display = "none";
    document.querySelector(".loader").style.display = "block";

    // Make prediction by calling api /predict
    axios.post("/predict", form_data)
      .then(function(response) {
        // Get and display the result
        document.querySelector(".loader").style.display = "none";
        document.querySelector("#result").style.display = "block";
        document.querySelector("#result").textContent = " Resultado: " + response.data;
        console.log("¡Éxito!");
      })
      .catch(function(error) {
        // Handle error
        console.log(error);
      });
  });
});
