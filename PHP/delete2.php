<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Specify the file paths to be deleted
    $uploadFolder = 'upload/';
    $frameFolder = 'frame/';
    $graphFolder = 'graph/';
    $tempframefolder= 'temp_frame/';
    // Delete all files in the upload folder
    $filesInUploadFolder = glob($uploadFolder . '*');
    foreach ($filesInUploadFolder as $file) {
        if (is_file($file)) {
            unlink($file);
        }
    }

    // Delete all files in the graph folder
    $filesInUploadFolder = glob($graphFolder . '*');
    foreach ($filesInUploadFolder as $file) {
        if (is_file($file)) {
            unlink($file);
        }
    }

    // Delete all subfolders inside the frame folder
    deleteSubfolders($frameFolder);
    deleteSubfolders($tempframefolder);

    echo 'Files and subfolders deleted successfully.';
} else {
    echo 'Invalid request.';
}

function deleteSubfolders($folderPath) {
    if (is_dir($folderPath)) {
        $subfolders = glob($folderPath . '/*', GLOB_ONLYDIR);
        foreach ($subfolders as $subfolder) {
            deleteFolder($subfolder);
        }
    }
}

function deleteFolder($folderPath) {
    if (is_dir($folderPath)) {
        $files = glob($folderPath . '/*');
        foreach ($files as $file) {
            is_dir($file) ? deleteFolder($file) : unlink($file);
        }
        rmdir($folderPath);
    }
}
?>
