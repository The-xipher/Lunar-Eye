$(document).ready(function() {
    // ajax call for already existing email verification
    $("#stuemail").on("keypress blur", function() {
        var reg = /^[A-Z0-9._%+-]+@([A-Z0-9-]+\.)+[A-Z]{2,4}$/i;
        var stuemail = $("#stuemail").val();
      $.ajax({
        url: 'student/addmember.php',
        method: 'POST',
        data: {
          checkemail: "checkemail",
          stuemail: stuemail,
        },
        success: function(data) {
        //   console.log(data);
        if(data!=0){
            $("#statusmsg2").html("<small style='color:red;'>    Email Already Exist</small>");
            $("#signup").attr("disabled",true)
        }
        else if(data==0 && reg.test(stuemail) )
        {
            $("#statusmsg2").html("<small style='color:green;'>&nbsp;&nbsp;There You Go!..</small>");
            $("#signup").attr("disabled",false)
        }
        else if(stuemail=="")
        {
            $("#statusmsg2").html("<small style='color:red;'>    Please Enter a Valid Email</small>"); 
        }

        else if (!reg.test(stuemail)) {
            $("#statusmsg2").html("<small style='color:red;'>&nbsp;&nbsp;Please Enter Valid Email");
            $("#signup").attr("disabled",true)
        }
        if(stuemail=="")
        {
            $("#statusmsg2").html("<small style='color:red;'>    Please Enter Valid Email</small>"); 
        }
        },
      });
    });
  });


function addstu() {
    var reg = /^[A-Z0-9._%+-]+@([A-Z0-9-]+\.)+[A-Z]{3,4}$/i;
    var stuname = $("#stuname").val();
    var stuemail = $("#stuemail").val();
    var stupass = $("#stupass").val();

    if (stuname.trim() == "") {
        $("#statusmsg1").html("<small style='color:red;'>    Please Enter Your Name</small>");
        $("#stuname").focus();
        return false;
    } else if (stuemail.trim() == "") {
        $("#statusmsg2").html("<small style='color:red;'>    Please Enter Your Email</small>");
        $("#stuemail").focus();
        return false;
    } 
    else if (stuemail.trim() != "" && !reg.test(stuemail)) {
        $("#statusmsg2").html("<small style='color:red;'>    Please Enter Valid Email</small>");
        $("#stuemail").focus();
        return false;
    } 
    else if (stupass.trim() == "") {
        $("#statusmsg3").html("<small style='color:red;'>    Please Enter Your Password</small>");
        $("#stupass").focus();
        return false;
    } 
    else {
        $.ajax({
            url: 'student/addmember.php',
            method: 'POST',
            dataType: "json",
            data: {
                stusignup: "stusignup",
                stuname: stuname,
                stuemail: stuemail,
                stupass: stupass,
            },
            success: function (data) {
                console.log(data)
                if (data == "OK") {
                    $('#successmsg').html("<span class='alert alert-success'>Registration Successful! Please Close and Login </span>");
                    clearStuRegField()
                } else if (data == "Fail") {
                    $('#successmsg').html("<span class='alert alert-danger'>Unable to Register!</span>");
                }
            },
        });
    }
    


} 
// Empty All Fields

function clearStuRegField() {

    $("#stuRegForm").trigger("reset");
    
    $("#statusMsg1").html(" ");
    
    $("#statusMsg2").html(" ");
    
    $("#statusMsg3").html(" ");
    
    }
