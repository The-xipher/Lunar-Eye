<!DOCTYPE html>
<html lang="en">
<head>
    <title>Image Upload</title>
    <link rel="stylesheet" href="style.css">
    <style>
        #loadingContainer {
            display: none;
            margin-top: 10px;
        }

        #loadingGif {
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
<h1>Image Upload</h1>
<div class="container">

    <form method="post" action="upload.php" enctype="multipart/form-data" id="uploadForm">
        <label for="imageInput" id="selectImageLabel">Select an Image</label>
        <input type="file" name="imageInput" id="imageInput" accept="image/*" onchange="displayImage()" style="display: none;">
        <div id="imagePreviewContainer" style="display: none;">
            <img id="imagePreview" alt="Selected Image">
        </div>
        <button type="button" id="uploadBtn" style="display: none;" onclick="uploadImage()">Upload Image</button>
        <div id="loadingContainer" style="display: none;">
            <img id="loadingGif" src="image/loading.gif" alt="Loading..."> <!-- Replace 'loading.gif' with the path to your loading GIF -->
        </div>
        <div id="messageContainer"></div>
    </form>

    <button type="button" id="refreshBtn" onclick="refresh()">Refresh</button>
    <button type="button" id="returnHomeBtn" onclick="returnHome()">Home</button>

<script>
    function displayImage() {
        var input = document.getElementById('imageInput');
        var previewContainer = document.getElementById('imagePreviewContainer');
        var preview = document.getElementById('imagePreview');

        var file = input.files[0];

        if (file) {
            var reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                document.getElementById('uploadBtn').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    function displayLoading() {
        document.getElementById('loadingContainer').style.display = 'block';
        document.getElementById('uploadBtn').style.display = 'none';
        document.getElementById('imageInput').style.display = 'none'; // Hide the Select an Image input
        document.getElementById('selectImageLabel').style.display = 'none'; // Hide the Select an Image label
    }

    function hideLoading() {
        document.getElementById('loadingContainer').style.display = 'none';
    }

    function uploadImage() {
        displayLoading(); // Show loading GIF

        var form = document.getElementById('uploadForm');
        var formData = new FormData(form);

        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'upload.php', true);

        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // Display the message in the message container
                document.getElementById('messageContainer').innerHTML = '<p>' + xhr.responseText + '</p>';
                hideLoading(); // Hide loading GIF
                document.getElementById('uploadBtn').style.display = 'none'; // Hide the Upload Image button
            }
        };

        xhr.send(formData);
    }

    function refresh() {
        // No need to send the fileName parameter as we want to delete all files
        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'delete.php', true);
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
        xhr.open('POST', 'delete.php', true);
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


    document.querySelector('label[for="imageInput"]').addEventListener('click', function (event) {
        event.preventDefault(); // Prevent the default behavior to avoid double-clicking issue
        document.getElementById('imageInput').click();
    });
</script>

</body>
</html>
