<?php
// Start a session
session_start();

// Destroy the session to log out the admin
session_destroy();

// Redirect to the index.php page
header("Location: index.php");
exit();
?>

