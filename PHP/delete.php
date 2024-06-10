<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $uploadFolder = 'data/Images/';

    // Delete all files in the upload folder
    $filesInUploadFolder = glob($uploadFolder . '*');
    foreach ($filesInUploadFolder as $file) {
        if (is_file($file)) {
            unlink($file);
        }
    }
} else {
    echo 'Invalid request.';
}
?>
