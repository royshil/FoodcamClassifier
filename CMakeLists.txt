PROJECT( foodcam_classifier )
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( OpenCV 2.3 REQUIRED )
message(${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE( manual-classifier manual_classifier.cpp )
TARGET_LINK_LIBRARIES( manual-classifier ${OpenCV_LIBS} )

ADD_EXECUTABLE( train-bovw train_bovw.cpp )
TARGET_LINK_LIBRARIES( train-bovw ${OpenCV_LIBS} )

ADD_EXECUTABLE( build-vocabulary build_vocabolary.cpp )
TARGET_LINK_LIBRARIES( build-vocabulary ${OpenCV_LIBS} )
