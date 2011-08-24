PROJECT( foodcam_classifier )
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( OpenCV 2.3 REQUIRED )
message(${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS -fopenmp )

ADD_EXECUTABLE( manual-classifier manual_classifier.cpp )
TARGET_LINK_LIBRARIES( manual-classifier ${OpenCV_LIBS} )

ADD_EXECUTABLE( train-bovw train_bovw.cpp training_common.cpp )
TARGET_LINK_LIBRARIES( train-bovw ${OpenCV_LIBS} )

ADD_EXECUTABLE( build-vocabulary build_vocabolary.cpp )
TARGET_LINK_LIBRARIES( build-vocabulary ${OpenCV_LIBS} )

ADD_EXECUTABLE( kmeans-trainer kmeans_trainer.cpp )
TARGET_LINK_LIBRARIES( kmeans-trainer ${OpenCV_LIBS} )

ADD_EXECUTABLE( make-test-background make_test_background_image.cpp )
TARGET_LINK_LIBRARIES( make-test-background ${OpenCV_LIBS} )

ADD_EXECUTABLE( test-classifiers test_classifiers.cpp predict_common.cpp )
TARGET_LINK_LIBRARIES( test-classifiers ${OpenCV_LIBS} )

ADD_EXECUTABLE( train-SVM-alone train_SVM_alone.cpp training_common.cpp )
TARGET_LINK_LIBRARIES( train-SVM-alone ${OpenCV_LIBS} )

ADD_EXECUTABLE( foodcam-predict foodcam_predict.cpp predict_common.cpp )
TARGET_LINK_LIBRARIES( foodcam-predict ${OpenCV_LIBS} )
