void guisetup()
{
  double knob = 0.5;
  
  TCanvas *c1 = new TCanvas("c1");
  c1->Divide(1,2);
  
  c1->cd(2);
  TSlider *slider = new TSlider("slider", "x", 0.1, 0.02, 0.98, 0.08);
  slider->SetMethod(".x reload.C");
  slider->SetRange(0,.9);
    
  c1->cd(1);
  TF1 *f = new TF1("f1","sin(x)",0,1);
  f->Draw();
}
