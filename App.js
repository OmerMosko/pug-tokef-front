import { View, Text } from "react-native";

export default function App() {
  console.log("App Exectued")
  let x = 1;
  x.toString()
  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Text>Universal Omer React with Expo</Text>
    </View>
  );
}
