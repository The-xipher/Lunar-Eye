<!DOCTYPE html>
<html>
<head>
    <title>View Members</title>
    <style>
        #video-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            object-fit: cover;
        }

        .container {
            background-color: transparent;
        }

        .search {
            margin-bottom: 20px;
        }

        .members-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: rgba(0, 0, 0, 0.7);
        }

        .members-table th,
        .members-table td {
            padding: 10px;
            text-align: left;
            color: white;
        }

        .members-table th {
            background-color: transparent;
        }

        .members-table tr:nth-child(even) {
            background-color: transparent;
        }

        .text-center .btn {
            background-color: black;
            color: white;
            padding: 10px 20px;
            border: 2px solid #4CAF50;
            border-radius: 30px;
            cursor: pointer;
        }

        .text-center .btn:hover {
            background-color: #222222;
            border-color: #45a049;
            color: white;
        }
    </style>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<video id="video-background" autoplay muted loop>
    <source src="video/details.mp4" type="video/mp4">
</video>
<audio autoplay loop src="video/music.mp3"></audio>
<div class="container" style="background-color: transparent;">
    <h1 class="text-center" style="color: white; margin-top: 10px;">View Members</h1>
    <div class="search">
        <form method="get">
            <div class="input-group">
                <input type="text" class="form-control" name="search" placeholder="Search Members" value="<?php echo isset($_GET['search']) ? $_GET['search'] : ''; ?>">
                <div class="input-group-append">
                    <button type="submit" class="btn btn-success">Search</button>
                </div>
            </div>
        </form>
    </div>
    <?php
    // Database connection code
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "lmsnew";

    // Create a new connection
    $conn = new mysqli($servername, $username, $password, $dbname);

    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // Search query
    $search = isset($_GET['search']) ? $_GET['search'] : '';
    $search = strtolower($search); // Convert search input to lowercase
    $sql = "SELECT * FROM memberadd WHERE LOWER(stu_name) LIKE '%$search%' OR LOWER(stu_email) LIKE '%$search%'";

    // Execute the query
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // Display the members in a table
        echo '<table class="members-table">';
        echo '<tr><th>Name</th><th>Email</th><th>Reserved Books</th><th>Actions</th></tr>';

        while ($row = $result->fetch_assoc()) {
            echo '<tr>';
            echo '<td>' . $row['stu_name'] . '</td>';
            echo '<td>' . $row['stu_email'] . '</td>';

            // Retrieve reserved books for the current user from bookreserve table
            $userEmail = $row['stu_email'];
            $reservedBooksSql = "SELECT title FROM bookreserve WHERE stu_email = '$userEmail'";
            $reservedBooksResult = $conn->query($reservedBooksSql);

            echo '<td>';
            if ($reservedBooksResult->num_rows > 0) {
                echo '<ul>';
                while ($reservedBookRow = $reservedBooksResult->fetch_assoc()) {
                    echo '<li>' . $reservedBookRow['title'] . '</li>';
                }
                echo '</ul>';
            } else {
                echo '<div class="alert alert-info">No reserved books.</div>';
            }
            echo '</td>';

            echo '<td>';
            echo '<a href="edit_members.php?id=' . $row['id'] . '" class="btn btn-dark">Update</a>';
            echo ' <a href="delete_member.php?id=' . $row['id'] . '" class="btn btn-danger" onclick="return confirmDelete()">Delete</a>';
            echo ' <a href="reservation_details.php?id=' . $row['id'] . '" class="btn btn-success">Reserved Details</a>';
            echo '</td>';

            echo '</tr>';
        }

        echo '</table>';
    } else {
        echo '<div class="alert alert-info">No members found.</div>';
    }

    // Close the database connection
    $conn->close();
    ?>
</div>
<div class="text-center">
    <a href="admin.php" class="btn btn-dark mt-2">Return to Admin Page</a>
</div>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
<script>
    function confirmDelete() {
        return confirm("Do you wish to delete this member?");
    }
</script>
</body>
</html>
