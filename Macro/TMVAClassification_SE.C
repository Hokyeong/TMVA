#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include <TMVA/DataLoader.h>
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int TMVAClassification_SE(TString myMethodList = "")
{

   // This loads the library
   TMVA::Tools::Instance();

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassification_SE" << std::endl;

   // Register the training and test trees
   // Monte Carlo - Signal Sample data input
   TFile* signalfile = new TFile("/home/hokyeong/Work/SingleElectron/Data/Sig/Sig_Region1.root");
   TTree* signaltree = (TTree*)signalfile->Get("t");
   std::cout << "--- TMVAClassification   : Using as signal file: " << signalfile->GetName() << std::endl;
   
   // Monte Carlo - Background Sample data input (Minimum-Bias sample)
   TFile* datafile = new TFile("/home/hokyeong/Work/SingleElectron/Data/Bkg/Bkg_Region1.root");
   TTree* datatree = (TTree*)datafile->Get("t");
   std::cout << "--- TMVAClassification   : Using as real data file: " <<datafile->GetName() << std::endl;  

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "output_SE.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   std::cout << "--- TMVAClassification   : Creating as resulting file: " << outputFile->GetName() << std::endl;

   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification_SE", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;G:AnalysisType=Classification" );

   TMVA::DataLoader *dataloader=new TMVA::DataLoader("Dataset_SE");

   //------------------------------------------------------------------------------------------------------------------------------------
   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
//   dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
//   dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
//   dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );

   dataloader->AddVariable( "ntEgEtIso", "ntEgEt", "units", 'F' );
   dataloader->AddVariable( "ntEgEtaIso", "ntEgEta", "units", 'F' );
   dataloader->AddVariable( "ntEgPhiIso", "ntEgPhi", "units", 'F' );
   dataloader->AddVariable( "IsoValue", "Isolation_value", "units", 'F' );
   dataloader->AddVariable( "NumOfTrks", "NumberOfTracks", "units", 'I' );
   dataloader->AddVariable( "track_pT1", "track_pT1", "units", 'F' );
   dataloader->AddVariable( "track_pT2", "track_pT2", "units", 'F' );
   dataloader->AddVariable( "track_pT3", "track_pT3", "units", 'F' );
   dataloader->AddVariable( "track_pT4", "track_pT4", "units", 'F' );
   dataloader->AddVariable( "track_pT5", "track_pT5", "units", 'F' );
   dataloader->AddVariable( "track_pT6", "track_pT6", "units", 'F' );

   // You can add so-called "Spectator variables", which are not used in the MVA training,
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
   // input variables, the response values of all trained MVAs, and the spectator variables

//   dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
//   dataloader->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );


   //------------------------------------------------------------------------------------------------------------------------------------
   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;
  
   // You can add an arbitrary number of signal or background trees  //essential part of classification
   dataloader->AddSignalTree    ( signaltree,     signalWeight );
   dataloader->AddBackgroundTree( datatree,   backgroundWeight );
   //------------------------------------------------------------------------------------------------------------------------------------

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = "";
   TCut mycutb = "";
   // for example: TCut mycuts = "B_J_mass==3.097 || B_J_chi2>0.1 || mumNHits>5 || mupNHits>5"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";

   // Tell the dataloader how to use the training and testing events
   // To also specify the number of testing events, use:
   //dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
   //    "NSigTrain=250:NBkgTrain=40:NSigTest=48317:NBkgTest=44:SplitMode=Random:NormMode=NumEvents:!V" );
   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
       "nTrain_Signal=250:nTrain_Background=40:SplitMode=Random:NormMode=NumEvents:!V" );

   //------------------------------------------------------------------------------------------------------------------------------------
   // ### Book MVA methods 

   // BDT - Boosted Decision Tree
   // Adaptive Boost
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTA",
                           "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

   // Gradiant Boost
   //factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG",
   //    "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );

   // MLP - Multi Layer Perceptron
   // Normal 
   factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPS", "H:!V:NeuronType=sigmoid:VarTransform=N:NCycles=600:HiddenLayers=N+7:TestRate=5:!UseRegulator" );
   //factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPT", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+7:TestRate=5:!UseRegulator" );
   //factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPR", "H:!V:NeuronType=ReLU:VarTransform=N:NCycles=600:HiddenLayers=N+7:TestRate=5:!UseRegulator" );
   //Back propagation
   //factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=sigmoid:VarTransform=N:NCycles=600:HiddenLayers=N+7:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );

   //BFGS training with bayesian regulators
   //factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=60:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" );
   
   //SVM - Support Vector Machine
   factory->BookMethod( dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm");

//-----------------------------------------------------------------------------------------------------------------------------------------
   // Train MVAs using the set of training events
   factory->TrainAllMethods();
   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();
   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> TMVAClassification   : Creating as resulting root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification_SE is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}

int main( int argc, char** argv )
{
   // Select methods (don't look at this code - not of interest)
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   return TMVAClassification_SE(methodList);
}

   
