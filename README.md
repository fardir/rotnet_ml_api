# RotNet barcode ML API

## :rocket: How To Run

1. The app is combined with docker, make sure your environment support docker too.
1. Launch the app with a simple command ```docker-compose up --build -d```
1. Well done! You are ready to go :partying_face:

## :shamrock: References

1. <b>Convolutional Neural Network</b>
1. <b>Transfer Learning</b>

## :gift: Contributing and Publication

1. We couldn't wait your contribution. Please report the bugs by the issues
1. If you want to send a code. Please send your pull request to us, we would review your code immediately.

## :package: API Docs

1. /predict
   ```text
   HTTP Method : POST
   Query Params : none
   Authorization : none
   Request Body : 
   - Multipart-form
        - field : 
            - barcode_image: file
   Response Body : image with barcode string value
   ```