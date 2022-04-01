package com.go.plantd;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.go.plantd.ml.DiseaseDetection;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, demoTxt, classified, clickHere;
    ImageView imageView, arrowImage;
    Button picture;
    int imagesize = 224; // default image size

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        demoTxt = findViewById(R.id.demoText);
        clickHere = findViewById(R.id.click_here);
        arrowImage = findViewById(R.id.demoArrow);
        classified = findViewById(R.id.classified);

        demoTxt.setVisibility(View.VISIBLE);
        clickHere.setVisibility(View.GONE);
        arrowImage.setVisibility(View.VISIBLE);
        classified.setVisibility(View.GONE);
        result.setVisibility(View.GONE);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                //launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1 );
                } else {
                    //request camera permission if we don't have
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);

                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode==1 && resultCode == RESULT_OK){
            assert data != null;
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension=Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail( image, dimension, dimension);
            imageView.setImageBitmap(image);
            demoTxt.setVisibility(View. GONE);
            clickHere.setVisibility (View. VISIBLE);
            arrowImage.setVisibility (View. GONE);
            classified.setVisibility (View. VISIBLE);
            result.setVisibility (View.VISIBLE);
            image = Bitmap.createScaledBitmap(image, imagesize, imagesize, false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode ,resultCode, data );
    }

    private void classifyImage(Bitmap image){
    try {
        DiseaseDetection model = DiseaseDetection.newInstance(getApplicationContext());

        //create input for reference
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize( new int[]{1,224, 224,3}, DataType.FLOAT32);
        ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4 * imagesize * imagesize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        //get 1D array of 224 224 pixels in image
        int[] intValue = new int[imagesize * imagesize];
        image.getPixels(intValue, 0, image.getWidth(), 0, 0,image.getWidth(), image.getHeight());

        // iterate over pixels and extract R, G, values, add to bytebuffer
        int pixel = 0;
        for (int i = 0; i < imagesize; i++) {
            for (int j = 0; j < imagesize; j++) {
                int val = intValue[pixel++]; // RUB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((((val >> 8) & 0xFF) * (1.f / 255.f)));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        inputFeature0.loadBuffer(byteBuffer);
        //zun model interface and gets result
        DiseaseDetection.Outputs outputs = model.process (inputFeature0);
        TensorBuffer outputFeatures0 = outputs.getOutputFeature0AsTensorBuffer();
        float [] confidence=outputFeatures0.getFloatArray();
        // find the index of the class with the biggest confidence
        int maxPos = 0;
        float maxConfidence = 0;
        for (int i = 0; i < confidence.length;i++) {
            if (confidence[i] > maxConfidence) {

                maxConfidence = confidence[i];
                maxPos = i;
            }
        }
        String s= Float. toString(maxConfidence);
        String[] classes = {"Pepper bell \n Bacterial spot","Potato \n Early blight","Tomato \n Target Spot" };
        result.setText("Tomato Leaf Mold");
        result.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // to search the disease on internet
                startActivity(new Intent(Intent.ACTION_VIEW,
                        Uri.parse("http://www.google.com/search?q=" + result.getText()+" remedy")));
            }
        });
        model.close();
        }catch (IOException e) {
        // TODO Handle the exception
        }
    }
}