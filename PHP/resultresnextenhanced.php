<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- <link rel="stylesheet" href="style2.css"> -->
    <title>Result</title>
    <style>
        #background-video {
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;    
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 80%;
            margin: 50px auto;
            padding: 20px;
            color: white;
            background-color: rgba(0, 0, 0, 0.5); /* Black with 50% opacity */
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            overflow: hidden; /* Ensure the container wraps around the video */
        }

        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        p {
            margin-bottom: 10px;
        }
        #videoContainer {
            width: 50%; /* Adjust the width as needed */
            float: left;
            margin-right: 20px;
        }
        #videoOutput {
            width: 100%;
        }
        #outputContainer {
            width: 45%; /* Adjust the width as needed */
            float: left;
        }
        #graphContainer {
            width: 80%;
            margin: 50px auto;
            padding: 20px;
            color: white;
            background-color: rgba(0, 0, 0, 0.5); /* Black with 50% opacity */
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            overflow: hidden; /* Ensure the container wraps around the video */
        }
        #graphContainer img {
            display: block; /* Ensure the image takes up the full width of the container */
            margin: 0 auto; /* Center the image horizontally */
            max-width: 100%; /* Make sure the image doesn't exceed the container width */
        }
        #refreshBtn {
            position: fixed;
            top: 10px;
            left: 25px;
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        #returnHomeBtn {
            position: fixed;
            top: 10px;
            right: 40px;
            padding: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
<video autoplay muted loop id="background-video">
        <source src="video/result.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <div class="container">
        <h1>Result</h1>
        <div id="videoContainer">
            <video id="videoOutput" controls width="250" height="250">
                <?php
                // Scan the 'upload' directory and fetch the video file name
                $files = scandir('upload');
                foreach ($files as $file) {
                    // Ignore "." and ".." directories
                    if ($file != '.' && $file != '..') {
                        echo "<source src='upload/$file' type='video/mp4'>";
                    }
                }
                ?>
                Your browser does not support the video tag.
            </video>
        </div>
        <div id="outputContainer">
            <h2><strong>Output:</strong></h2>
            <?php
            // Get the output parameter from the URL
            if(isset($_GET['output'])) {
                $output = $_GET['output'];
                // Split the output by newline characters and display each line separately
                $lines = explode("\n", $output);
                foreach ($lines as $line) {
                    echo "<h3>$line</h3>";
                }
            } else {
                echo "<p>No output available.</p>";
            }
            ?>
        </div>
        <div style="clear:both;"></div> <!-- Clear float -->
    </div>
    <!-- New container for the graph -->
    <div id="graphContainer">
        <h1>Frame-Confidence Graph</h1>
        <img src="graph/confidence_graph.png" alt="Graph">
    </div>
    <button type="button" id="refreshBtn" onclick="refresh()">Back</button>
    <button type="button" id="returnHomeBtn" onclick="returnHome()">Home</button>
    <script>
        function refresh() {
            // No need to send the fileName parameter as we want to delete all files
            var xhr = new XMLHttpRequest();
            xhr.open('POST', 'delete2.php', true);
            xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');

            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    // Reload the page after successful deletion
                    window.location.href = 'resnextvideoenhanced.php';
                }
            };

            // No need to send any parameters
            xhr.send();
        }

        function returnHome() {
            // No need to send the fileName parameter as we want to delete all files
            var xhr = new XMLHttpRequest();
            xhr.open('POST', 'delete2.php', true);
            xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');

            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    // Reload the page after successful deletion
                    window.location.href = 'modelselect.php';
                }
            };

            // No need to send any parameters
            xhr.send();
        }
    </script>
</body>
</html>
