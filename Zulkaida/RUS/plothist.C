void plothist() {

char name[100];
double temp[18480];
ifstream inputfile;

sprintf(name,"HistoContent.txt");
inputfile.open(name,ios::in);
cout<<" Get the data from "<<name<<endl<<endl;

for (int i = 0; i < 6231 ; i++)
  {
    inputfile >> temp[i];
  }

TH2I *hist = new TH2I("hist","hist",31,0,31,201,0,201);
for (int j=0; j<201; j++)
  {
   for (int i=0; i<31; i++) {
    hist->SetBinContent(i+1,j+1, temp[i*201+j]);
   }
  }

TCanvas *c28 = new TCanvas("c28","c28"); //c0->SetGrid();
        c28->cd();
        hist->SetTitle("MC Hit-Matrix ; Det-Id; Ele-Id ");
        hist->SetStats(0);
        hist->Draw("COLZ");
        c28->SaveAs("plot.png");

}
