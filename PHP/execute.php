<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Get the uploaded image path from the command line argument
    $uploadedImagePath = "data/Images";

    if (!empty($uploadedImagePath)) {
        // Execute the Python script with the uploaded image path
        $output = shell_exec('python hello.py' . $uploadedImagePath); // Replace with the actual path to your Python script

        // Display only the last 4 characters of the entire output
        $lastFourCharacters = substr(trim($output), -4);
        echo "Last 4 characters of the entire output: $lastFourCharacters";
    } else {
        echo "Error: No image path provided.";
    }
}
?>
