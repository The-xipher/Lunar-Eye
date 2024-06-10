<!-- start registration -->
            <!-- Button trigger modal -->
        
        
        <!-- Modal -->
        <div class="modal fade" id="signupmodel" tabindex="-1" aria-labelledby="signupmodelLabel" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                <h1 class="modal-title fs-5" id="signupmodelLabel">User Registration </h1>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">

                    <!-- start sign up form -->
                    <form id="stuRegForm">
                    
                    <div class="form-group">
                    
                        <i class="fas fa-user"></i><label for="stuname" class="ps-2 font-weight-bold">Name </label><small id="statusmsg1"> </small><input type="text" class="form-control" placeholder="Name" name="stuname" id="stuname">
                    
                    </div>
                    <br>
                    <div class="form-group">
                    
                    <i class="fas fa-envelope"></i><label for="stuemail"
                    
                    class="ps-2 font-weight-bold">Email</label><small id="statusmsg2"> </small><input
                    
                    type="email"
                    
                    class="form-control" placeholder="Email" name="stuemail" id="stuemail">
                    
                    <small class="form-text">We'll never share your email with anyone else.</small>
                    
                    </div>
                    <br>
                    <div class="form-group">

                        <i class="fas fa-key"></i><label for="stupass" class="ps
                        -2 font-weight-bold">New
                        
                        Password</label><small id="statusmsg3"> </small><input type="password" class="form-control" placeholder="Password" name="stupass" id="stupass">
                        
                        </div>
                    
                        
                        </form>
                        <!-- end sign up form -->
                    </div>
                <div class="modal-footer">
                    <span id="successmsg"></span>
                <button type="button" class="btn btn-primary" onclick="addstu()" id="signup" >Sign Up</button>
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                
                </div>
            </div>
            </div>
        </div>
    <!-- END registration -->