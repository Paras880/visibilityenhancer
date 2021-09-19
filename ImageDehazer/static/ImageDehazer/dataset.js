$(function() {

    $('[data-toggle="modal"]').hover(function() {
      var modalId = $(this).data('target');
      $(modalId).modal('show');
  
    });
  
  });

  // var modal = document.getElementById('#basicExampleModal');

  // // Get the image and insert it inside the modal - use its "alt" text as a caption
  // var img = document.getElementById('myImg');
  // var modalImg = document.getElementById("img01");
  // var captionText = document.getElementById("caption");
  // img.onclick = function(){
  //     modal.style.display = "block";
  //     modalImg.src = this.src;
  //     captionText.innerHTML = this.alt;
  // }
