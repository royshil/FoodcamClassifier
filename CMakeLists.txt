PROJECT( foodcam_classifier )
FIND_PACKAGE( OpenCV REQUIRED )

ADD_EXECUTABLE( manual-classifier manual_classifier.cpp )
TARGET_LINK_LIBRARIES( manual-classifier ${OpenCV_LIBS} )

ADD_EXECUTABLE( train-bovw train_bovw.cpp )
TARGET_LINK_LIBRARIES( train-bovw ${OpenCV_LIBS} )

ADD_EXECUTABLE( build-vocabulary build_vocabulary.cpp )
TARGET_LINK_LIBRARIES( build-vocabulary ${OpenCV_LIBS} )