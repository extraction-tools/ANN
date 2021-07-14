#include <top/G4_Beamline.C>
#include <top/G4_Target.C>
#include <top/G4_InsensitiveVolumes.C>
#include <top/G4_SensitiveDetectors.C>

R__LOAD_LIBRARY(libfun4all)
R__LOAD_LIBRARY(libPHPythia8)
R__LOAD_LIBRARY(libg4detectors)
R__LOAD_LIBRARY(libg4testbench)
R__LOAD_LIBRARY(libg4eval)
R__LOAD_LIBRARY(libg4dst)
R__LOAD_LIBRARY(libdptrigger)
R__LOAD_LIBRARY(libembedding)
R__LOAD_LIBRARY(libevt_filter)
R__LOAD_LIBRARY(libktracker)
R__LOAD_LIBRARY(libSQPrimaryGen)
R__LOAD_LIBRARY(libana_trkqa)
R__LOAD_LIBRARY(libana_trkqa_v2)

using namespace std;

int Fun4Sim(const int nevent = 1234)
{
  const double target_coil_pos_z = -300;
  const int nmu = 1;
  int embedding_opt = 0;
  const bool legacy_rec_container = true;

  const bool do_collimator = true;
  const bool do_target = true;
  const bool do_e1039_shielding = true;
  const bool do_fmag = true;
  const bool do_kmag = true;
  const bool do_absorber = true;
  const bool do_dphodo = true;
  const bool do_station1DC = false;       //station-1 drift chamber should be turned off by default

  const double target_l = 7.9; //cm
  const double target_z = (7.9-target_l)/2.; //cm
  const int use_g4steps = 1;

  const double FMAGSTR = -1.054;
  const double KMAGSTR = -0.951;

  const bool gen_pythia8  = true; // false;
  const bool gen_cosmic   = false;
  const bool gen_gun      = false;
  const bool gen_particle = false;
  const bool read_hepmc   = false;
  const bool gen_e906legacy = false; //E906LegacyGen()
 // const bool save_in_acc  = false; //< Set true to save only in-acceptance events into DST.

 //! vtx gen flag
  const bool legacyVtxGen = true;

  recoConsts *rc = recoConsts::instance();
  rc->set_DoubleFlag("FMAGSTR", FMAGSTR);
  rc->set_DoubleFlag("KMAGSTR", KMAGSTR);
  if(gen_cosmic) {
    rc->init("cosmic");
    rc->set_BoolFlag("COARSE_MODE", true);
    rc->set_DoubleFlag("KMAGSTR", 0.);
    rc->set_DoubleFlag("FMAGSTR", 0.);
  }
  if(legacyVtxGen)
  {
    rc->set_BoolFlag("TARGETONLY", false);
    rc->set_BoolFlag("DUMPONLY", true);
  }
  rc->Print();

  GeomSvc::UseDbSvc(true);
  GeomSvc *geom_svc = GeomSvc::instance();
  //const double x0_shift = 0.0; //cm 
  //std::cout << "D2X::X0: " << geom_svc->getDetectorX0("D2X") << std::endl;
  //geom_svc->setDetectorX0("D2X", geom_svc->getDetectorX0("D2X")+x0_shift);
  //std::cout << "D2X::X0: " << geom_svc->getDetectorX0("D2X") << std::endl;

  ///////////////////////////////////////////
  // Make the Server
  //////////////////////////////////////////
  Fun4AllServer *se = Fun4AllServer::instance();
  se->Verbosity(0);


  // pythia8
  if(gen_pythia8) {    
    PHPythia8 *pythia8 = new PHPythia8();
    //pythia8->Verbosity(99);
    pythia8->set_config_file("phpythia8_DY.cfg");
    //pythia8->set_config_file("phpythia8_Jpsi.cfg");
    if(legacyVtxGen) pythia8->enableLegacyVtxGen();
    else {
    pythia8->set_vertex_distribution_mean(0, 0, target_coil_pos_z, 0);
    pythia8->set_vertex_distribution_function(PHHepMCGenHelper::Uniform,

                                              PHHepMCGenHelper::Uniform,

                                              PHHepMCGenHelper::Uniform,

                                              PHHepMCGenHelper::Uniform);

    pythia8->set_vertex_distribution_width(0,0,4.,0);
    }
    pythia8->set_embedding_id(1);
    se->registerSubsystem(pythia8);

    pythia8->set_trigger_AND();

    PHPy8ParticleTrigger* trigger_mup = new PHPy8ParticleTrigger();
    trigger_mup->AddParticles("-13");
    //trigger_mup->SetPxHighLow(7, 0.5);
    //trigger_mup->SetPyHighLow(6, -6);
    trigger_mup->SetPzHighLow(120, 10);
    pythia8->register_trigger(trigger_mup);

    PHPy8ParticleTrigger* trigger_mum = new PHPy8ParticleTrigger();
    trigger_mum->AddParticles("13");
    //trigger_mum->SetPxHighLow(-0.5, 7);
    //trigger_mum->SetPyHighLow(6, -6);
    trigger_mum->SetPzHighLow(120, 10);
    pythia8->register_trigger(trigger_mum);
  }
  
  if(gen_pythia8 || read_hepmc) {
    HepMCNodeReader *hr = new HepMCNodeReader();
    hr->set_particle_filter_on(true);
    hr->insert_particle_filter_pid(13);
    hr->insert_particle_filter_pid(-13);
    se->registerSubsystem(hr);
  }

  // single gun
  if(gen_gun) {
    PHG4ParticleGun *gun = new PHG4ParticleGun("GUN");
    gun->set_name("mu-");
    //gun->set_vtx(0, 0, target_coil_pos_z);
    //gun->set_mom(3, 3, 50);
    gun->set_vtx(30, 10, 590);
    gun->set_mom(-0.3, 2, 50);
    se->registerSubsystem(gun);
  }

  // multi particle gun
  if(gen_particle) {
    PHG4SimpleEventGenerator *genp = new PHG4SimpleEventGenerator("MUP");
    //genp->set_seed(123);
    genp->add_particles("mu+", nmu);  // mu+,e+,proton,pi+,Upsilon
    genp->set_vertex_distribution_function(PHG4SimpleEventGenerator::Uniform,
        PHG4SimpleEventGenerator::Uniform,
        PHG4SimpleEventGenerator::Uniform);
    genp->set_vertex_distribution_mean(0.0, 0.0, target_coil_pos_z);
    genp->set_vertex_distribution_width(0.0, 0.0, 0.0);
    genp->set_vertex_size_function(PHG4SimpleEventGenerator::Uniform);
    genp->set_vertex_size_parameters(0.0, 0.0);

    if(FMAGSTR>0)
      //genp->set_pxpypz_range(0,6, -6,6, 10,100);
      genp->set_pxpypz_range(-3,6, -3,3, 10,100);
    else
      //genp->set_pxpypz_range(-6,0, -6,6, 10,100);
      genp->set_pxpypz_range(-6,3, -3,3, 10,100);


    genp->Verbosity(0);
    se->registerSubsystem(genp);
  }

  if(gen_particle) {
    PHG4SimpleEventGenerator *genm = new PHG4SimpleEventGenerator("MUP");
    //genm->set_seed(123);
    genm->add_particles("mu-", nmu);  // mu+,e+,proton,pi+,Upsilon
    genm->set_vertex_distribution_function(PHG4SimpleEventGenerator::Uniform,
        PHG4SimpleEventGenerator::Uniform,
        PHG4SimpleEventGenerator::Uniform);
    genm->set_vertex_distribution_mean(0.0, 0.0, target_coil_pos_z);
    genm->set_vertex_distribution_width(0.0, 0.0, 0.0);
    genm->set_vertex_size_function(PHG4SimpleEventGenerator::Uniform);
    genm->set_vertex_size_parameters(0.0, 0.0);

    if(FMAGSTR>0)
      //genm->set_pxpypz_range(-6,0, -6,6, 10,100);
      genm->set_pxpypz_range(-6,3, -3,3, 10,100);
    else
      //genm->set_pxpypz_range(0,6, -6,6, 10,100);
      genm->set_pxpypz_range(-3,6, -3,3, 10,100);

    genm->Verbosity(0);
    se->registerSubsystem(genm);
  }

  // E906LegacyGen
  if(gen_e906legacy){
    SQPrimaryParticleGen *e906legacy = new  SQPrimaryParticleGen();
    
    const bool pythia_gen = false;
    const bool drellyan_gen = true;
    const bool JPsi_gen = false;
    const bool Psip_gen = false;  

    if(drellyan_gen){
      e906legacy->set_xfRange(0.1, 0.5); //[-1.,1.]
      e906legacy->set_massRange(0.23, 10.0);// 0.22 and above     
      e906legacy->enableDrellYanGen();
    }
   
   
    if(Psip_gen){ 
      e906legacy->set_xfRange(0.1, 0.5); //[-1.,1.]
      e906legacy->enablePsipGen();
    }


    if(JPsi_gen){
      e906legacy->set_xfRange(0.1, 0.5); //[-1.,1.]
      e906legacy->enableJPsiGen();
    }
    
    if(pythia_gen) e906legacy->enablePythia();

    se->registerSubsystem(e906legacy);
  }

  if(gen_cosmic) {
    SQCosmicGen* cosmicGen = new SQCosmicGen();
    se->registerSubsystem(cosmicGen);
  }

  // Fun4All G4 module
  PHG4Reco *g4Reco = new PHG4Reco();
  //PHG4Reco::G4Seed(123);
  //g4Reco->set_field(5.);
  g4Reco->set_field_map(
      rc->get_CharFlag("fMagFile")+" "+
      rc->get_CharFlag("kMagFile")+" "+
      Form("%f",FMAGSTR) + " " +
      Form("%f",KMAGSTR) + " " +
      "5.0",
      PHFieldConfig::RegionalConst);
  // size of the world - every detector has to fit in here
  g4Reco->SetWorldSizeX(1000);
  g4Reco->SetWorldSizeY(1000);
  g4Reco->SetWorldSizeZ(5000);
  // shape of our world - it is a tube
  g4Reco->SetWorldShape("G4BOX");
  // this is what our world is filled with
  g4Reco->SetWorldMaterial("G4_AIR"); //G4_Galactic, G4_AIR
  // Geant4 Physics list to use
  g4Reco->SetPhysicsList("FTFP_BERT");

  // insensitive elements of the spectrometer
  SetupInsensitiveVolumes(g4Reco, do_e1039_shielding, do_fmag, do_kmag, do_absorber);

  // collimator, targer and shielding between target and FMag
  SetupBeamline(g4Reco, do_collimator, target_coil_pos_z - 302.36); // Is the position correct??

  if (do_target) {
    SetupTarget(g4Reco, target_coil_pos_z, target_l, target_z, use_g4steps);
  }

  // sensitive elements of the spectrometer
  SetupSensitiveDetectors(g4Reco, do_dphodo, do_station1DC);

  se->registerSubsystem(g4Reco);

  
  // save truth info to the Node Tree
  PHG4TruthSubsystem *truth = new PHG4TruthSubsystem();
  g4Reco->registerSubsystem(truth);

  // digitizer
  SQDigitizer *digitizer = new SQDigitizer("DPDigitizer", 0);
  //digitizer->Verbosity(99);
  digitizer->set_enable_st1dc(do_station1DC);    // these two lines need to be in sync with the parameters used
  digitizer->set_enable_dphodo(do_dphodo);       // in the SetupSensitiveVolumes() function call above
  se->registerSubsystem(digitizer);

  /// Save only events that are in the geometric acceptance.
  SQGeomAcc* geom_acc = new SQGeomAcc();
  geom_acc->SetMuonMode(SQGeomAcc::PAIR); // PAIR, PAIR_TBBT, SINGLE, SINGLE_T, etc.
  geom_acc->SetPlaneMode(SQGeomAcc::HODO_CHAM); // HODO, CHAM or HODO_CHAM
  geom_acc->SetNumOfH1EdgeElementsExcluded(4); // Exclude 4 elements at H1 edges
  se->registerSubsystem(geom_acc); 

  // Make SQ nodes for truth info
  se->registerSubsystem(new TruthNodeMaker());

  // embedding
  if(embedding_opt == 1) {
    SRawEventEmbed *embed = new SRawEventEmbed("SRawEventEmbed");
    embed->set_in_name("digit_016070_R007.root");
    embed->set_in_tree_name("save");
    embed->set_trigger_bit((1<<0));
    //embed->set_in_name("random_run3a_1.root");
    //embed->set_in_tree_name("mb");
    //embed->set_trigger_bit((1<<7));
    embed->Verbosity(0);
    se->registerSubsystem(embed);
  }

  // Trigger Emulator
  DPTriggerAnalyzer* dptrigger = new DPTriggerAnalyzer();
  dptrigger->set_road_set_file_name("$E1039_RESOURCE/trigger/trigger_67.txt");
  se->registerSubsystem(dptrigger);

  // Event Filter
  //EvtFilter *evt_filter = new EvtFilter();
  //evt_filter->Verbosity(10);
  //evt_filter->set_trigger_req(1<<5);
  //se->registerSubsystem(evt_filter);

  AnaTrkQAv2* trackQAv2 = new AnaTrkQAv2();
  trackQAv2->set_out_name("trackQA_v2.root");
  se->registerSubsystem(trackQAv2);

  // Tracking module
  SQReco* reco = new SQReco();
  reco->Verbosity(0);
  //reco->set_geom_file_name("support/geom.root"); //not needed as it's created on the fly
  reco->set_enable_KF(true);           //Kalman filter not needed for the track finding, disabling KF saves a lot of initialization time
  reco->setInputTy(SQReco::E1039);     //options are SQReco::E906 and SQReco::E1039
  reco->setFitterTy(SQReco::KFREF);    //not relavant for the track finding
  reco->set_evt_reducer_opt("none");   //if not provided, event reducer will be using JobOptsSvc to intialize; to turn off, set it to "none", for normal tracking, set to something like "aoc"
  reco->set_enable_eval(true);          //set to true to generate evaluation file which includes final track candidates 
  reco->set_eval_file_name("eval.root");
  reco->set_enable_eval_dst(false);     //set to true to include final track cnadidates in the DST tree
  if(gen_cosmic) reco->add_eval_list(3);    //output of cosmic reco is contained in the eval output for now
  //reco->add_eval_list(3);             //include back partial tracks in eval tree for debuging
  //reco->add_eval_list(2);             //include station-3+/- in eval tree for debuging
  //reco->add_eval_list(1);             //include station-2 in eval tree for debugging
  se->registerSubsystem(reco);

  VertexFit* vertexing = new VertexFit();
  //vertexing->enable_fit_target_center(); //uncomment if you want to fit in the target center
  //vertexing->enableOptimization(); //uncomment if you want to fit according to new optimization formula
  se->registerSubsystem(vertexing);

  //Analysis module for track QA
  AnaTrkQA* trackQA = new AnaTrkQA();
  trackQA->set_out_name("trackQA.root");
  se->registerSubsystem(trackQA); 

  //AnaTrkQAv2* trackQAv2 = new AnaTrkQAv2();
  //trackQAv2->set_out_name("trackQA_v2.root");
  //se->registerSubsystem(trackQAv2);  

  //// Trim minor data nodes (to reduce the DST file size)
  //se->registerSubsystem(new SimDstTrimmer());

  // input - we need a dummy to drive the event loop
  if(read_hepmc) {
    Fun4AllHepMCInputManager *in = new Fun4AllHepMCInputManager("HEPMCIN");
    in->Verbosity(10);
    in->set_vertex_distribution_mean(0,0,target_coil_pos_z,0);
    se->registerInputManager(in);
    in->fileopen("hepmcout.txt");
  } else {
    Fun4AllInputManager *in = new Fun4AllDummyInputManager("DUMMY");
    se->registerInputManager(in);
  }

  ///////////////////////////////////////////
  // Output
  ///////////////////////////////////////////

  // DST output manager
  //Fun4AllDstOutputManager *out = new Fun4AllDstOutputManager("DSTOUT", "DST.root");
  //se->registerOutputManager(out);

  //if(gen_pythia8 && !read_hepmc) {
  //  Fun4AllHepMCOutputManager *out = new Fun4AllHepMCOutputManager("HEPMCOUT", "hepmcout.txt");
  //  out->set_embedding_id(1);
  //  se->registerOutputManager(out);
  //}

  se->run(nevent);

  PHGeomUtility::ExportGeomtry(se->topNode(),"geom.root");
  
  // finish job - close and save output files
  se->End();
  se->PrintTimer();
  rc->WriteToFile("recoConsts.tsv");
  std::cout << "All done" << std::endl;

  // cleanup - delete the server and exit
  delete se;
  gSystem->Exit(0);
  return 0;
}
