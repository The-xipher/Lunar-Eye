function adminlogin() {
    console.log("Clicked");
    var adminlogemail = $("#adminlogemail").val();
    var adminlogpass = $("#adminlogpass").val();
    $.ajax({
        url: 'student/adminloginset.php',
        method: 'POST',
        data: {
            checkadminemail: "checkadminemail",
            adminlogemail: adminlogemail,
            adminlogpass: adminlogpass,
        },
        success: function (data) {
            if (data == 0) {
                $("#adminlogmsg").html('<small class="alert alert-danger">Invalid Email-id or Password</small>');
            } else if (data == 1) {
                $("#adminlogmsg").html('<div class="spinner-border text-success" role="status"></div>');
                setTimeout(() => {
                    adminlogin();
                    window.location.href="admin.php";
                }, 1000);
            }
            
        },
    });
}