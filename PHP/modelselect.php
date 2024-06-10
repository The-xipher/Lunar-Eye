<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="css/style2.css">
    <style>
        /* Custom styling for the logout button */
        .logout-button {
            background-color: #4a5568; /* Example background color */
            color: #fff; /* Example text color */
            padding: 10px 20px; /* Example padding */
            border: none; /* Remove default border */
            border-radius: 5px; /* Example border radius */
            cursor: pointer; /* Add pointer cursor on hover */
            transition: background-color 0.3s; /* Smooth transition on hover */
            position: absolute; /* Position absolutely within parent */
            bottom: 10px; /* Adjust bottom position */
            left: 50%; /* Align horizontally to center */
            transform: translateX(-50%); /* Center horizontally */
        }

        .logout-button:hover {
            background-color: #2d3748; /* Example background color on hover */
        }
    </style>
</head>
<body>
    <video autoplay muted loop id="background-video">
        <source src="video/b&w.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <!-- <?php
    session_start();
    $db_host = "localhost";
    $db_user = "root";
    $db_password = "";
    $db_name = "lmsnew";

    // Create Connection
    $conn = new mysqli($db_host, $db_user, $db_password, $db_name);

    // Check Connection
    if($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    
    if(isset($_SESSION['userEmail'])) {
        $userEmail = $_SESSION['userEmail'];
        
        // Fetch the user's name from memberadd table using the email
        $sql = "SELECT stu_name FROM memberadd WHERE stu_email='$userEmail'";
        $result = $conn->query($sql);
        if ($result->num_rows > 0) {
            $row = $result->fetch_assoc();
            $userName = $row['stu_name'];
            
            echo '<h1 class="heading" style="color: white;"">Welcome, ' . $userName . '</h1>';
        } else {
            echo '<h1 class="heading" style="color: white;"">Welcome, User</h1>'; // Default welcome message if user's name is not found
        }
    } else {
        echo '<h1 class="heading" style="color: white;"">Welcome, User</h1>'; // Default welcome message if user is not logged in
    }
    ?> -->
    <div class="title">
        <h1>Model Selection</h1>
        <p>Choose the Required Model</p>
    </div>

    <div class="flip-card-container">
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="image/resnext.jpg" alt="" class="card-image">
                </div>
                <div class="flip-card-back">
                    <div class="back-header">
                        <img src="image/pic3.jpg" alt="">
                    </div>
                    <div class="back-footer">
                        <h2>Resnext</h2>
                        <p>Resnext Deepfake Detection Model</p>
                        <a href="resnextvideo.php">Base</a>
                        <a href="resnextvideoenhanced.php">Enhanced</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="image/mesonet.jpg" alt="" class="card-image">
                </div>
                <div class="flip-card-back">
                    <div class="back-header">
                        <img src="image/pic2.jpg" alt="">
                    </div>
                    <div class="back-footer">
                        <h2>Mesonet</h2>
                        <p>Mesonet Deepfake Detection Model</p>
                        <a href="mesonetimg.php">Image</a>
                        <a href="mesonetvideo.php">Video</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="image/combined.jpg" alt="" class="card-image">
                </div>
                <div class="flip-card-back">
                    <div class="back-header">
                        <img src="image/pic1.jpg" alt="">
                    </div>
                    <div class="back-footer">
                        <h2>Ensemble Model</h2>
                        <p>Resnext and Mesonet Combined Model</p>
                        <a href="ensemblenormal.php">Base</a>
                        <a href="ensembleenhanced.php">Enhanced</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Logout button without Tailwind CSS -->
    <form method="post" action="logout.php" class="text-center">
        <input type="submit" value="Logout" class="logout-button">
    </form>
</body>
</html>
