import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../DB/DatabaseHelper.dart';
import '../HomeScreen.dart';
import 'Recognition.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  static const int WIDTH = 160;
  static const int HEIGHT = 160;
  final dbHelper = DatabaseHelper();
  Map<String,Recognition> registered = Map();
  @override
  String get modelName => 'assets/facenet.tflite';
  // @override
  // String get modelName => 'assets/mobile_face_net.tflite';

  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    loadModel();
    initDB();
  }

  initDB() async {
    await dbHelper.init();
    loadRegisteredFaces();
  }

  void loadRegisteredFaces() async {
    final allRows = await dbHelper.queryAllRows();
   // debugPrint('query all rows:');
    for (final row in allRows) {
    //  debugPrint(row.toString());
      print(row[DatabaseHelper.columnName]);
      String name = row[DatabaseHelper.columnName];
      List<double> embd = row[DatabaseHelper.columnEmbedding].split(',').map((e) => double.parse(e)).toList().cast<double>();
      Recognition recognition = Recognition(row[DatabaseHelper.columnName],Rect.zero,embd,0);
      registered.putIfAbsent(name, () => recognition);
    }
  }

  // void registerFaceInDB(String name, String embedding) async {
  //   // row to insert
  //   Map<String, dynamic> row = {
  //     DatabaseHelper.columnName: name,
  //     DatabaseHelper.columnEmbedding: embedding
  //   };
  //   final id = await dbHelper.insert(row);
  //   print('inserted row id: $id');
  // }

  void registerFaceInDB(String name, List<double> embedding) async {
    // row to insert
    Map<String, dynamic> row = {
      DatabaseHelper.columnName: name,
      DatabaseHelper.columnEmbedding: embedding.join(",")
    };
    final id = await dbHelper.insert(row);
    print('inserted row id: $id');
    loadRegisteredFaces();
  }


  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(modelName);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  List<dynamic> imageToArray(img.Image inputImage){
    img.Image resizedImage = img.copyResize(inputImage!, width: WIDTH, height: HEIGHT);
    // List<double> flattenedList = resizedImage.data!.expand((channel) => [channel.r, channel.g, channel.b]).map((value) => value.toDouble()).toList();
    // Initialize list to store flattened pixel data
    List<double> flattenedList = [];

    // Loop through each pixel in the resized image and extract RGB values
    for (int i = 0; i < resizedImage.length; i++) {
      int pixel = resizedImage[i];

      // Extract red, green, and blue channels
      int red = img.getRed(pixel);
      int green = img.getGreen(pixel);
      int blue = img.getBlue(pixel);

      // Convert to double and add to flattened list
      flattenedList.addAll([red.toDouble(), green.toDouble(), blue.toDouble()]);
    }
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] = float32Array[c * height * width + h * width + w];
        }
      }
    }
    return reshapedArray.reshape([1,160,160,3]);
  }

  Recognition recognize(img.Image image,Rect location) {

    //TODO crop face from image resize it and convert it to float array
    var input = imageToArray(image);
    print(input.shape.toString());

    //TODO output array
    List output = List.filled(1*512, 0).reshape([1,512]);

    //TODO performs inference
    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final run = DateTime.now().millisecondsSinceEpoch - runs;
    // print('Time to run inference: $run ms$output');

    //TODO convert dynamic list to double list
     List<double> outputArray = output.first.cast<double>();

     //TODO looks for the nearest embeeding in the database and returns the pair
     Pair pair = findNearest(outputArray);
     print("distance= ${pair.distance}");

     return Recognition(pair.name,location,outputArray,pair.distance);
  }

  //TODO  looks for the nearest embeeding in the database and returns the pair which contain information of registered face with which face is most similar
  findNearest(List<double> emb){
    Pair pair = Pair("Unknown", -5);
    for (MapEntry<String, Recognition> item in registered.entries) {
      final String name = item.key;
      List<double> knownEmb = item.value.embeddings;
      double distance = 0;
      for (int i = 0; i < emb.length; i++) {
        double diff = emb[i] -
            knownEmb[i];
        distance += diff*diff;
      }
      distance = sqrt(distance);
      if (pair.distance == -5 || distance < pair.distance) {
        pair.distance = distance;
        pair.name = name;
      }
    }
    return pair;
  }
  // double cosineDistance(List<double> emb1, List<double> emb2) {
  //   double dotProduct = 0.0;
  //   double norm1 = 0.0;
  //   double norm2 = 0.0;
  //
  //   for (int i = 0; i < emb1.length; i++) {
  //     dotProduct += emb1[i] * emb2[i];
  //     norm1 += emb1[i] * emb1[i];
  //     norm2 += emb2[i] * emb2[i];
  //   }
  //
  //   return 1 - (dotProduct / (sqrt(norm1) * sqrt(norm2)));
  // }
  //
  // findClose(List<double> emb) {
  //   Pair pair = Pair("Unknown", double.infinity); // Bắt đầu với khoảng cách vô cực
  //   for (MapEntry<String, Recognition> item in registered.entries) {
  //     final String name = item.key;
  //     List<double> knownEmb = item.value.embeddings;
  //
  //     // Tính khoảng cách Euclid
  //     double euclidDistance = 0;
  //     for (int i = 0; i < emb.length; i++) {
  //       double diff = emb[i] - knownEmb[i];
  //       euclidDistance += diff * diff; // Bình phương độ chênh lệch
  //     }
  //     euclidDistance = sqrt(euclidDistance); // Lấy căn bậc hai
  //
  //     // Tính khoảng cách Cosine
  //     double dotProduct = 0;
  //     double normA = 0;
  //     double normB = 0;
  //     for (int i = 0; i < emb.length; i++) {
  //       dotProduct += emb[i] * knownEmb[i];
  //       normA += emb[i] * emb[i];
  //       normB += knownEmb[i] * knownEmb[i];
  //     }
  //     normA = sqrt(normA);
  //     normB = sqrt(normB);
  //     double cosineSimilarity = dotProduct / (normA * normB);
  //     double cosineDistance = 1 - cosineSimilarity;
  //
  //     // Tính khoảng cách Manhattan
  //     double manhattanDistance = 0;
  //     for (int i = 0; i < emb.length; i++) {
  //       manhattanDistance += (emb[i] - knownEmb[i]).abs(); // Tính tổng độ chênh lệch tuyệt đối
  //     }
  //
  //     // Kết hợp các khoảng cách
  //     double combinxedDistance = (euclidDistance + cosineDistance + manhattanDistance) / 3; // Hoặc dùng trọng số nếu cần
  //
  //     if (combinedDistance < pair.distance) {
  //       pair.distance = combinedDistance; // Cập nhật khoảng cách nhỏ nhất
  //       pair.name = name; // Cập nhật tên
  //     }
  //   }
  //   return pair; // Trả về cặp tên và khoảng cách nhỏ nhất
  // }


  void close() {
    interpreter.close();
  }

}
class Pair{
   String name;
   double distance;
   Pair(this.name,this.distance);
}


