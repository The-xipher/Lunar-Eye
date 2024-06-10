<?php
include('./header.php');

?>
    <!-- video start -->

    <div class="container-fluid remove-margin">

        <div class="vid-parent">

            <video playsinline autoplay muted loop >

                <source src="video/eye.mp4" type="">

            </video>
        <div class="vid-overlay"></div>
        </div>
        <div class="vid-content">
            <h1 class='my-content'> Welcome to Lunar Eye</h1>
            <br>
            <a href="#" class="btn btn-light" data-bs-toggle="modal" data-bs-target="#signupmodel">Get Started</a> 
        </div>

    </div>
    <!-- video end -->

    <!-- start text banner --
    <div class="container-fluid  bg-dark txt-banner">

        <div class="row bottom-banner">

            <div class="col-sm">

                <h5><i class="fa-solid fa-globe"></i> Online Access</h5>

            </div>

            <div class="col-sm">

                <h5><i class="fa-solid fa-magnifying-glass"></i> Search Facility</h5>

            </div>

            <div class="col-sm">
                
                <h5><i class="fa-solid fa-book"></i> Library Administration</h5>

            </div>
    
            <div class="col-sm">
                
                <h5><i class="fa-solid fa-people-group"></i> Community Administration</h5>

            </div>
        </div>
    </div>
    !-- end text banner -->



    





   <?php
   include('./footer.php');
   include('./loginmember.php');
   include('./adminlogin.php');
   include('./signup.php');
   ?>
   <script src="js/ajaxrequest(signup/verification).js"> </script>
   <script src="js/ajaxadminlogin.js"> </script>
   <script src="js/ajaxmemberlogin.js"> </script>
   <script src="js/popper.min.js"> </script>
    <script src="js/bootstrap.min.js"> </script>
    <script src="js/all.min.js"> </script>

    
    

    

</body>
</html>






