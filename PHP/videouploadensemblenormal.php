<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Check if the file is uploaded successfully
    if ($_FILES['videoInput']['error'] == UPLOAD_ERR_OK) {
        // Specify the target directory for file upload
        $targetDirectory = "upload/";

        // Get the uploaded file information
        $uploadedFileName = basename($_FILES['videoInput']['name']);
        $targetFilePath = $targetDirectory . $uploadedFileName;

        // Move the uploaded file to the target directory
        if (move_uploaded_file($_FILES['videoInput']['tmp_name'], $targetFilePath)) {
            $output = shell_exec('python ensemblenormal.py'); // Replace with the actual path to your Python script
            
            // Explode the output string by new line characters and filter only lines starting with "print"
            $lines = explode("\n", trim($output));
            foreach ($lines as $line) {
                if (strpos($line, '*') === 0) {
                    echo $line . "<br>";
                }
            }
        } else {
            echo "<p>Error moving the uploaded image to the target directory.</p>";
        }
    } else {
        echo "<p>Error uploading the image. Error code: {$_FILES['videoInput']['error']}</p>";
    }
}   
?>
