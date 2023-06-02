void reload()
{
  
  Double_t binxmin = slider->GetMinimum();
  Double_t binxmax = slider->GetMaximum();
  
  
  cout << "Binxmin: " <<binxmin<<endl;
  cout << "Binxmax: " <<binxmax<<endl;
  
  c1->Update();
}
