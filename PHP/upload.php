<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Check if the file is uploaded successfully
    if ($_FILES['imageInput']['error'] == UPLOAD_ERR_OK) {
        // Specify the target directory for file upload
        $targetDirectory = 'data/Images/';

        // Get the uploaded file information
        $uploadedFileName = basename($_FILES['imageInput']['name']);
        $targetFilePath = $targetDirectory . $uploadedFileName;

        // Move the uploaded file to the target directory
        if (move_uploaded_file($_FILES['imageInput']['tmp_name'], $targetFilePath)) {
            $output = shell_exec('python script.py '); // Replace with the actual path to your Python script
            $lastFourCharacters = substr(trim($output), -13);
            echo $lastFourCharacters;
        } else {
            echo "<p>Error moving the uploaded image to the target directory.</p>";
        }
    } else {
        echo "<p>Error uploading the image. Error code: {$_FILES['imageInput']['error']}</p>";
    }
}
?>
