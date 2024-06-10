<!DOCTYPE html>
<html lang="en">
<head>
    <title>Video Upload</title>
    <link rel="stylesheet" href="style2.css">
    <style>
        #loadingContainer1 {
            display: none;
            margin-top: 10px;
        }

        #loadingGif1 {
            width: 200px; /* Adjust the width as needed */
            height: 100px; /* Adjust the height as needed */
        }
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        h1 {
            background-color: rgba(0, 0, 0, 0.5);
            padding-top: 20px;
            text-align: center;
            color: white;
        }
    </style>
</head>
<body>
<video id="backgroundVideo" autoplay muted loop>
    <source src="video/4K .mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<h1>Enhanced Resnext Video Upload</h1>

<div class="container">
    <form method="post" action="videouploadresnextenhanced.php" enctype="multipart/form-data" id="videouploadForm">
        <label for="videoInput" id="selectVideoLabel">Select a Video</label>
        <input type="file" name="videoInput" id="videoInput" accept="video/*" onchange="displayVideo()" style="display: none;">
        <div id="VideoPreviewContainer" style="display: none;">
            <video id="VideoPreview" controls width="100%">
                <source src="" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <button type="button" id="uploadBtn1" style="display: none;" onclick="uploadVideo()">Upload Video</button>
        <div id="loadingContainer1" style="display: none;">
            <img id="loadingGif1" src="image/loading.gif" alt="Loading..."> <!-- Replace 'loading.gif' with the path to your loading GIF -->
        </div>
        <div id="messageContainer"></div>
    </form>

    <button type="button" id="refreshBtn" onclick="refresh()">Refresh</button>
    <button type="button" id="returnHomeBtn" onclick="returnHome()">Home</button>
</div>

<script>
    function displayVideo() {
        var input = document.getElementById('videoInput');
        var previewContainer = document.getElementById('VideoPreviewContainer');
        var preview = document.getElementById('VideoPreview');

        var file = input.files[0];

        if (file) {
            var reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                document.getElementById('uploadBtn1').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    function displayLoading1() {
        document.getElementById('loadingContainer1').style.display = 'block';
        document.getElementById('uploadBtn1').style.display = 'none';
        document.getElementById('videoInput').style.display = 'none'; // Hide the Select a Video input
        document.getElementById('selectVideoLabel').style.display = 'none'; // Hide the Select a Video label
    }

    function hideLoading1() {
        document.getElementById('loadingContainer1').style.display = 'none';
    }

    function uploadVideo() {
        displayLoading1(); // Show loading GIF

        var form = document.getElementById('videouploadForm');
        var formData = new FormData(form);

        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'videouploadresnextenhanced.php', true);

        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // Redirect to result.php with the output as parameter
                window.location.href = 'resultresnextenhanced.php?output=' + encodeURIComponent(xhr.responseText);
            }
        };

        xhr.send(formData);
    }

    function refresh() {
        // No need to send the fileName parameter as we want to delete all files
        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'delete2.php', true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');

        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // Reload the page after successful deletion
                window.location.reload(true);
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
