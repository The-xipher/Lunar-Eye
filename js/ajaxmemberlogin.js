// code1
function login() {
    console.log("Clicked");
    var stulogemail = $("#useremail").val();
    var stulogpass = $("#userpass").val();
    $.ajax({
        url: 'student/memberlogin.php',
        method: 'POST',
        data: {
            checklogemail: "checklogemail",
            stulogemail: stulogemail,
            stulogpass: stulogpass,
        },
        success: function (data) {
            if (data == 0) {
                $("#statuslogmsg").html('<small class="alert alert-danger">Invalid Email-id or Password</small>');
            } else if (data == 1) {
                $("#statuslogmsg").html('<div class="spinner-border text-success" role="status"></div>');
                // Store the email in localStorage for later use
                localStorage.setItem('userEmail', stulogemail);
                setTimeout(() => {
                    window.location.href = "modelselect.php";
                }, 1000);
            }
        },
    });
}
