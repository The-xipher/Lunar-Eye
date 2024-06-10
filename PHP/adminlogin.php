
<!-- start admin login registration -->
            <!-- Button trigger modal -->
        
        
        <!-- Modal -->
        <div class="modal fade" id="adminloginmodel" tabindex="-1" aria-labelledby="adminloginmodelLabel" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                <h1 class="modal-title fs-5" id="adminloginmodelLabel"> Admin Login</h1>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    

                    <!-- start sign up form -->
                    <form>
                    
                        <div class="form-group">
                        
                        <i class="fas fa-envelope"></i><label for="adminlogemail"
                        
                        class="ps-2 font-weight-bold">Email</label><input
                        
                        type="email"
                        
                        class="form-control" placeholder="Email" name="adminlogemail" id="adminlogemail">
                        
                       
                        
                        </div>
                        <br>
                        <div class="form-group">
    
                            <i class="fas fa-key"></i><label for="adminlogpass" class="ps
                            -2 font-weight-bold">Password</label><input type="password" class="form-control" placeholder="Password" name="adminlogpass" id="adminlogpass">
                            
                            </div>
                    
                            </form>
                            <!-- end sign up form -->
                            <div class="modal-footer">
                            <small id="adminlogmsg"></small>
                                <button type="button" class="btn btn-primary" id="adminloginbtn" onclick="adminlogin()">Login</button>
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                
                                </div>
                
                </div>
            </div>
            </div>
        </div>
    <!-- END admin login registration -->