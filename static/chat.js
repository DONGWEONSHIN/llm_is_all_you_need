/*
function upload() {
    const fileupload = $("#fileupload")[0];
  console.debug();
    // 파일을 여러개 선택할 수 있으므로 files 라는 객체에 담긴다.
    console.log("fileupload: ", fileupload.files)

    if(fileupload.files.length === 0){
      alert("파일은 선택해주세요");
      return;
    }

    const file = new FormData();
    file.append("file", fileupload.files[0]);
    
    // file 이름 저장
    filename = fileupload.files[0].name;

    $.ajax({
      type:"POST",
      url: "/savePdf",
      processData: false,
      contentType: false,
      data: file,
    }).done(function(data) {
      if (data.RETURN_FLAG === "SUCCESS") {
        alert("파일이 성공적으로 저장되었습니다.");
      } else {
        alert("파일이 저장되지 않았습니다.");
      }
    });
}
*/

$(document).ready(function() {                      // .ready document가 실행되면 한 번만 로드되는 함수
  $("#messageArea").on("submit", function(event) {  // .on id가 messageArea의 form 태그에 submit 이벤트가 발생하면 function() 함수를 실행
      const date = new Date();                      // 날짜
      const hour = date.getHours();                 // 시간
      const minute = date.getMinutes();             // 분
      const str_time = hour+":"+minute;             // 시간 : 분
      var rawText = $("#text").val();               // #text 추출

      var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_cotainer_send">' + rawText + '<span class="msg_time_send">'+ str_time + '</span></div><div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg"></div></div>';
      
      $("#text").val("");
      $("#messageFormeight").append(userHtml);      // .append #messageFormeight에 userHTML을 추가


      $.ajax({                                      // ajax로 서버와 통신
          data: {                                   // data안에 msg : rawText를 담는 기능
              msg: rawText,	
          },
          type: "POST",
          url: "/chat",
      }).done(function(data) {                      // .done 서버로부터 응답이 성공적으로 도착하면 실행되는 함수
          var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">' + data + '<span class="msg_time">' + str_time + '</span></div></div>';
          $("#messageFormeight").append($.parseHTML(botHtml));    // #messageFormeight에 botHtml을 추가 
      });                                                         // parseHTML은 문자열로 표현된 HTML을 DOM요소로 변환 -> 변환된 Html 요소를 #messageFormeight에 추가
      event.preventDefault();                       // form 태그는 기본적으로 페이지 새로고침을 하지만 이것을 방지
    });

    //
    $("#uploadForm").on("submit", function(event) {
    
      const fileupload = $("#fileupload")[0];
      console.debug();
        // 파일을 여러개 선택할 수 있으므로 files 라는 객체에 담긴다.
        console.log("fileupload: ", fileupload.files)
    
        if(fileupload.files.length === 0){
          alert("파일은 선택해주세요");
          return;
        }
    
        const file = new FormData();
        file.append("file", fileupload.files[0]);
        
        // file 이름 저장
        filename = fileupload.files[0].name;
    
        $.ajax({
          type:"POST",
          url: "/savePdf",
          processData: false,
          contentType: false,
          data: file,
        }).done(function(data) {
          if (data.RETURN_FLAG === "SUCCESS") {
            alert("파일이 성공적으로 저장되었습니다.");
          } else {
            alert("파일이 저장되지 않았습니다.");
          }
        });
    });
});

