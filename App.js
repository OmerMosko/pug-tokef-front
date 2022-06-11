import { StyleSheet, Text, View, SafeAreaView, Button, Image } from 'react-native';
import { useEffect, useRef, useState } from 'react';
import { Camera } from 'expo-camera';
import { shareAsync } from 'expo-sharing';
import { Video } from 'expo-av';
import * as MediaLibrary from 'expo-media-library';
import * as Speech from 'expo-speech';
import React from 'react';

export default function App() {
  // Camera
  let cameraRef = useRef();
  const [hasCameraPermission, setHasCameraPermission] = useState();
  const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState();
  const [hasMicrophonePermission, setHasMicrophonePermission] = useState();

  const [photo, setPhoto] = useState();

  const [isRecording, setIsRecording] = useState(false);
  const [video, setVideo] = useState();

  const listAllVoiceOptions = async () => {
    let voices = await Speech.getAvailableVoicesAsync();
    // console.log(voices);
  };
  
  React.useEffect(listAllVoiceOptions);
  // Speech 
  const speakGreeting = (name) => {
    var greeting = "21.12.2022";
    const options = {
      voice: "com.apple.speech.synthesis.voice.Fred",
    };
    Speech.speak(greeting, options)
  };  
  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermissionsAsync();
      console.log(cameraPermission)
      setHasCameraPermission(cameraPermission.status === "granted");
      const mediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();
      console.log(mediaLibraryPermission)
      setHasMediaLibraryPermission(mediaLibraryPermission.status === "granted");
      const microphonePermission = await Camera.requestMicrophonePermissionsAsync();
      setHasMicrophonePermission(microphonePermission.status === "granted");

    })();
    
  }, []);

  if (hasCameraPermission === undefined) {
    return <Text>Requesting permissions...</Text>
  } else if (!hasCameraPermission) {
    return <Text>Permission for camera not granted. Please change this in settings.</Text>
  }

  let takePic = async () => {
    let options = {
      quality: 1,
      base64: true,
      exif: false
    };
    console.log('Capturing');

    let newPhoto = await cameraRef.current.takePictureAsync(options);
    setPhoto(newPhoto);
    MediaLibrary.saveToLibraryAsync(newPhoto.uri).then(() => {
      const form = new FormData();

      form.append('image', {
        uri: newPhoto.uri,
        type: 'image/jpg',
        name: 'image.jpg',
      });

      fetch('http://89.139.10.158:5000/uploader', {
        method: 'POST',
        headers: {
          contentType: "text/html; charset=utf-8",
        },
        body: form
      })
      .then((response) => {
        console.log("recived")
        console.log(response)
        speakGreeting(response)
      }).catch((error)=>{
        console.error("Failed uploader")
      });
      setPhoto(undefined);
    });
    
   
  };

  let recordVideo = () => {
    setIsRecording(true);
    let options = {
      quality: "1080p",
      maxDuration: 5,
      mute: false
    };

    cameraRef.current.recordAsync(options).then((recordedVideo) => {
      setVideo(recordedVideo);
      
      MediaLibrary.saveToLibraryAsync(recordedVideo.uri).then(() => {
        

        const form = new FormData();

        form.append('image', {
          uri: recordedVideo.uri,
          type: 'image/jpg',
          name: 'image.jpg',
        });

        fetch('http://89.139.10.158:5000/uploader', {
          method: 'POST',
          headers: {
            contentType: "text/html; charset=utf-8",
          },
          body: form
        })
        .then((response) => {
          console.log("recived")
          console.log(response)
          speakGreeting(response)
        }).catch((error)=>{
          console.error("Failed uploader")
        });
      

        setVideo(undefined);
      });
      
      setIsRecording(false);
    });
  };

  let stopRecording = () => {
    setIsRecording(false);
    cameraRef.current.stopRecording();
  };
  let saveVideo = () => {
    
  };
  if (video) {
    let shareVideo = () => {
      shareAsync(video.uri).then(() => {
        setVideo(undefined);
      });
    };

    

    // return (
    //   <SafeAreaView style={styles.container}>
    //     <Video
    //       style={styles.video}
    //       source={{uri: video.uri}}
    //       useNativeControls
    //       resizeMode='contain'
    //       isLooping
    //     />
    //     <Button title="Share" onPress={shareVideo} />
    //     {hasMediaLibraryPermission ? <Button title="Save" onPress={saveVideo} /> : undefined}
    //     <Button title="Discard" onPress={() => setVideo(undefined)} />
    //   </SafeAreaView>
    // );
  }

  if (photo) {
    let sharePic = () => {
      shareAsync(photo.uri).then(() => {
        setPhoto(undefined);
      });
    };

    let savePhoto = () => {
      
    };

    return (
      <SafeAreaView style={styles.container}>
        <Image style={styles.preview} source={{ uri: "data:image/jpg;base64," + photo.base64 }} />
        <Button title="Share" onPress={sharePic} />
        {hasMediaLibraryPermission ? <Button title="Save" onPress={savePhoto} /> : undefined}
        <Button title="Discard" onPress={() => setPhoto(undefined)} />
      </SafeAreaView>
    );
  }

  return (
    <>
    <Camera style={styles.container} ref={cameraRef}>
      <View style={styles.buttonContainer}>
          <Button title="TakePic" onPress={takePic} />
          <Button title={isRecording ? "Stop Recording" : "Record Video"} onPress={isRecording ? stopRecording : recordVideo} />
      </View>
    </Camera>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: "150%",
  },
  buttonContainer: {
    backgroundColor: '#fff',
  },
  preview: {
    alignSelf: 'stretch',
  },
  video: {
    flex: 1,
    alignSelf: "stretch"
  }
});