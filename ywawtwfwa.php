<?php
session_start();

    include("config.php");
    include("function.php");
    use GuzzleHttp\Client;

    $user_data = check_login($connection);

    //display profile infos
    $id = $_SESSION["user_id"];
    $result = mysqli_query($connection, "SELECT * FROM booking WHERE user_id = $id");
    $row = mysqli_fetch_assoc($result);

    $query = "SELECT venuename FROM venues WHERE venuetype = 'Wedding' ";
    $queryres = mysqli_query($connection, $query);

    $query2 = "SELECT theme_name FROM theme WHERE theme_type = 'Wedding' ";
    $themeres = mysqli_query($connection, $query2);

    //unique id
    $checkid = "SELECT id FROM wedding ORDER BY id DESC LIMIT 1"; // select only id column
    $checkresult = mysqli_query($connection, $checkid);

        if(mysqli_num_rows($checkresult) > 0)
        {

        $row = mysqli_fetch_assoc($checkresult);
        $uid = $row['id'];
        $get_numbers = str_replace("WED-", "", $uid);
        $id_increase = $get_numbers+1;
        $get_string = str_pad($id_increase, 4, 0, STR_PAD_LEFT);
        $newid = "WED-".$get_string; 

        } else {
            $newid = "WED-0001";
        }

    if(isset($_POST['submit']))
    {

        //// API endpoint URL
        $api_url = 'http://127.0.0.1:5000/predict';
        //http://127.0.0.1:5000

        $id = $_SESSION['user_id'];
        //$booking_id = 
        $firstname = $_POST['firstname'];
        $lastname = $_POST['lastname'];
        $phone = $_POST['phone'];
        $address =  $_POST['address'];
        $emailadd = $_POST['emailadd'];
        $bookdate = $_POST['bookdate'];

        $firstname = ucwords($firstname);
        $lastname = ucwords($lastname);
        $address = ucwords($address);

        $dayOfWeek = date('N', strtotime($bookdate));

        if ($dayOfWeek == 6 || $dayOfWeek == 7) {
            $weektype = 'weekend';
        } else {
            $weektype = 'weekday';
        }

        //if date is not at least one month from today
        $current_date = date('Y-m-d'); // Get the current date in YYYY-MM-DD format
        $one_month_from_now = strtotime('+1 month', strtotime($current_date));
        $user_date = strtotime($bookdate);

        //if date is already booked
        $query3 = "SELECT * FROM booking WHERE bookdate='$bookdate' AND status = 'confirmed' ";
        $dateres = mysqli_query($connection, $query3);

        if(strtotime($bookdate) < time()) {
            echo "<script>alert('The date you submitted is already in the past. Please choose a new date or edit the existing date.');</script>";
            echo "<script>window.history.back();</script>";
        } elseif(mysqli_num_rows($dateres) > 0) {
            echo "<script>alert('Your preferred target date is already booked. Please choose other dates. Thank you');</script>";
            echo "<script>window.history.back();</script>";
        } elseif($user_date < $one_month_from_now) {
            echo "<script>alert('Please choose a date that is at least one month from today.');</script>";
            echo "<script>window.history.back();</script>";
        } else {
            //
                        if(empty($_POST['guests']) || empty($_POST['venue'])  || empty( $_POST['cuisine'])  || empty( $_POST['style']) || empty( $_POST['theme'])  || empty( $_POST['checkbox']))
                        {
                            ?><script type="text/javascript">
                            alert('Fill in all fields before submitting. Restart filling in the forms again.');
                            window.history.back();
                            </script>
                            <?php
                        } else {
                            $event = 'Wedding';
                            $venue = $_POST['venue'];
                            $guests = $_POST['guests'];
                            $cuisine = $_POST['cuisine'];
                            $style = $_POST['style'];
                            $theme = $_POST['theme'];
                            $message = $_POST['message'];
        
                            $checkbox = $_POST['checkbox'];
                            $services = implode(",", $checkbox);

                              // Create variables for each service and set their value to 1 if they were selected, or 0 otherwise
                            $dj = in_array('DJ Services', $checkbox) ? 1 : 0;
                            $emcee = in_array('Emcee', $checkbox) ? 1 : 0;
                            $photo = in_array('Photographer', $checkbox) ? 1 : 0;
                            $video = in_array('Videographer', $checkbox) ? 1 : 0;
                            $makeup = in_array('Makeup Artist', $checkbox) ? 1 : 0;
                            $bar = in_array('Bar Area', $checkbox) ? 1 : 0;
                            $invitation = in_array('Invitation Cards', $checkbox) ? 1 : 0;
                            $none = in_array('None', $checkbox) ? 1 : 0;

                            echo $event;
                            echo $venue;
                            echo $cuisine;
                            echo $style;
                            echo $guests;
                            echo $weektype;
                            echo $dj;
                            echo $emcee;
                            echo $photo;
                            echo $video;
                            echo $makeup;
                            echo $bar;
                            echo $invitation;
                            echo $none;

                            // Input data
                            $input_data = array(
                                'event' => $event,
                                'venue' => $venue,
                                'cuisine' => $cuisine,
                                'style' => $style,
                                'guest_number' => $guests,
                                'weektype' => $weektype,
                                'dj_services' => $dj,
                                'emcee' => $emcee,
                                'photog' => $photo,
                                'videog' => $video,
                                'm_artist' => $makeup,
                                'bar_area' => $bar,
                                'inv_cards' => $invitation
                            );

                            /*
                            if (count($input_data) > 0) {
                                // The array has elements inside it.
                                echo 'The array has elements inside it.';
                              } else {
                                // The array is empty.
                                echo 'The array is empty.';
                              }
                              */


                            // Send POST request to the API
                            $ch = curl_init($api_url);
                            curl_setopt($ch, CURLOPT_POST, true);
                            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($input_data));
                            curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
                            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
                            $response = curl_exec($ch);
                            curl_close($ch);

                            // Check if API request was successful
                            if ($response === false) {
                                die('Error occurred while calling the API.');
                            }

                            // Parse the API response
                            //$data = json_decode($response->getBody(), true);
                            //$prediction = $data["Prediction"];
                            $result = json_decode($response, true);

                            //$predicted_price = $result['predicted_price'];
                            //echo $predicted_price;

                            // Check if prediction is available
                            
                            if (isset($result['prediction'])) {
                                $prediction = $result['prediction'];
                                echo 'Prediction: ' . $prediction;
                            } else {
                                echo 'Prediction not available.';
                            }
        
                            /*
                            $sql_booking = "INSERT INTO booking (user_id, booking_id, firstname, lastname, phone, address, emailadd, bookdate, weektype, event, status, coordinator_1, coordinator_2, coordinator_3, coordinator_4) VALUES ('$id', '$newid', '$firstname', '$lastname', '$phone', '$address', '$emailadd', '$bookdate', '$weektype', 'Wedding', 'pending', 'Pending', 'Pending', 'Pending', 'Pending' )";
                            $result = mysqli_query($connection, $sql_booking);
        
                            $sql_booking2 = "INSERT INTO wedding (user_id, booking_id, venue, guest_number, cuisine, style, theme_design, extra_services, other_preferences, dj_services, emcee, photographer, videographer, makeup_artist, bar_area, invitation_cards, none) VALUES ('$id', '$newid', '$venue', '$guests', '$cuisine', '$style', '$theme', '$services', '$message', '$dj', '$emcee', '$photo', '$video', '$makeup', '$bar','$invitation','$none')";
                            $result2 = mysqli_query($connection, $sql_booking2);
        
                            if($result && $result2)
                            {
                                ?><script type="text/javascript">
                                alert('Booked successfully.');
                                window.location.href='weddingbook.php';
                                </script>
                                <?php
                            } else {
                                //die(mysqli_error($connection));
                                ?><script type="text/javascript">
                                alert('Something went wrong. Please try again.');
                                </script>
                                <?php
                            }
                            */
                        }
                    } //deletable
            }


            
        


?>






















if(isset($_POST['submit']))
    {
        //// API endpoint URL
        $api_url = 'http://127.0.0.1:5000/predict';

        $event = $_POST['event'];
        $venue = $_POST['venue'];
        $guests = $_POST['guests'];
        $cuisine = $_POST['cuisine'];
        $style = $_POST['style'];
        
        $checkbox = $_POST['checkbox'];
        $services = implode(",", $checkbox);

        // Create variables for each service and set their value to 1 if they were selected, or 0 otherwise
        $dj = in_array('DJ Services', $checkbox) ? 1 : 0;
        $emcee = in_array('Emcee', $checkbox) ? 1 : 0;
        $photo = in_array('Photographer', $checkbox) ? 1 : 0;
        $video = in_array('Videographer', $checkbox) ? 1 : 0;
        $makeup = in_array('Makeup Artist', $checkbox) ? 1 : 0;
        $bar = in_array('Bar Area', $checkbox) ? 1 : 0;
        $invitation = in_array('Invitation Cards', $checkbox) ? 1 : 0;

        echo $event;
                            echo $venue;
                            echo $cuisine;
                            echo $style;
                            echo $guests;
                            echo $weektype;
                            echo $dj;
                            echo $emcee;
                            echo $photo;
                            echo $video;
                            echo $makeup;
                            echo $bar;
                            echo $invitation;

        // Input data
        $input_data = array(
            'event' => $event,
            'venue' => $venue,
            'cuisine' => $cuisine,
            'style' => $style,
            'guest_number' => $guests,
            'weektype' => $weektype,
            'dj_services' => $dj,
            'emcee' => $emcee,
            'photog' => $photo,
            'videog' => $video,
            'm_artist' => $makeup,
            'bar_area' => $bar,
            'inv_cards' => $invitation
        );

        // Send POST request to the API
        $ch = curl_init($api_url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($input_data));
        curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        $response = curl_exec($ch);
        curl_close($ch);

        // Check if API request was successful
        if ($response === false) {
            die('Error occurred while calling the API.');
        }

        // Parse the API response
        $result = json_decode($response, true);

        // Check if prediction is available
        if (isset($result['prediction'])) {
            $prediction = $result['prediction'];
            echo 'PHP ' . $prediction;
        } else {
            echo 'Prediction not available.';
        }



    }