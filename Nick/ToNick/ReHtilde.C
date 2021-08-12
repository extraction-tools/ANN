void ReHtilde()
{
//=========Macro generated from canvas: c4/ReHtilde
//=========  (Sat Jul 24 21:02:39 2021) by ROOT version 6.22/06
   TCanvas *c4 = new TCanvas("c4", "ReHtilde",4446,369,1685,899);
   c4->Range(-2.527348,-292.465,23.74296,325.4912);
   c4->SetFillColor(0);
   c4->SetBorderMode(0);
   c4->SetBorderSize(2);
   c4->SetFrameBorderMode(0);
   c4->SetFrameBorderMode(0);
   
   TMultiGraph *multigraph = new TMultiGraph();
   multigraph->SetName("");
   multigraph->SetTitle("ReHtilde; set;ReHtilde");
   
   Double_t local_fit_fx1001[20] = {
   1.05,
   2.05,
   3.05,
   4.05,
   5.05,
   6.05,
   7.05,
   8.05,
   9.05,
   10.05,
   11.05,
   12.05,
   13.05,
   14.05,
   15.05,
   16.05,
   17.05,
   18.05,
   19.05,
   20.05};
   Double_t local_fit_fy1001[20] = {
   22.9607,
   20.77331,
   -29.15593,
   -20.83674,
   -26.885,
   59.83735,
   -29.90802,
   49.63635,
   26.65161,
   28.32512,
   -104.4482,
   33.78785,
   -13.21956,
   -15.05717,
   -27.91037,
   14.13648,
   90.83165,
   169.7364,
   125.1907,
   17.78949};
   Double_t local_fit_fex1001[20] = {
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0};
   Double_t local_fit_fey1001[20] = {
   19.52755,
   15.63318,
   15.63365,
   18.23866,
   23.37475,
   65.45284,
   43.85958,
   40.98259,
   42.86286,
   49.39183,
   103.3969,
   33.62514,
   24.54953,
   25.37615,
   29.85049,
   68.44012,
   54.80166,
   71.84113,
   85.21122,
   104.3791};
   TGraphErrors *gre = new TGraphErrors(20,local_fit_fx1001,local_fit_fy1001,local_fit_fex1001,local_fit_fey1001);
   gre->SetName("local_fit");
   gre->SetTitle("");
   gre->SetFillStyle(1000);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#0000ff");
   gre->SetLineColor(ci);

   ci = TColor::GetColor("#0000ff");
   gre->SetMarkerColor(ci);
   gre->SetMarkerStyle(22);
   gre->SetMarkerSize(1.3);
   
   TH1F *Graph_local_fit1001 = new TH1F("Graph_local_fit1001","",100,0,21.95);
   Graph_local_fit1001->SetMinimum(-252.7874);
   Graph_local_fit1001->SetMaximum(286.5198);
   Graph_local_fit1001->SetDirectory(0);
   Graph_local_fit1001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_local_fit1001->SetLineColor(ci);
   Graph_local_fit1001->GetXaxis()->SetLabelFont(42);
   Graph_local_fit1001->GetXaxis()->SetTitleOffset(1);
   Graph_local_fit1001->GetXaxis()->SetTitleFont(42);
   Graph_local_fit1001->GetYaxis()->SetLabelFont(42);
   Graph_local_fit1001->GetYaxis()->SetTitleFont(42);
   Graph_local_fit1001->GetZaxis()->SetLabelFont(42);
   Graph_local_fit1001->GetZaxis()->SetTitleOffset(1);
   Graph_local_fit1001->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_local_fit1001);
   
   multigraph->Add(gre,"");
   
   Double_t simultaneous_fx1002[20] = {
   1.15,
   2.15,
   3.15,
   4.15,
   5.15,
   6.15,
   7.15,
   8.15,
   9.15,
   10.15,
   11.15,
   12.15,
   13.15,
   14.15,
   15.15,
   16.15,
   17.15,
   18.15,
   19.15,
   20.15};
   Double_t simultaneous_fy1002[20] = {
   0.253358,
   0.3513238,
   -0.5836429,
   -0.5187066,
   -0.686475,
   0.1968529,
   -0.1632258,
   0.3085937,
   0.2373978,
   0.3278268,
   -0.5619666,
   0.2404115,
   -0.1705604,
   -0.1440158,
   -0.2299514,
   0.02540333,
   0.6474852,
   2.993261,
   2.672178,
   0.4283671};
   Double_t simultaneous_fex1002[20] = {
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0};
   Double_t simultaneous_fey1002[20] = {
   1.83119,
   1.872768,
   2.160926,
   2.802767,
   3.994484,
   2.278937,
   2.039673,
   2.176432,
   2.507834,
   2.966607,
   6.544358,
   3.249964,
   3.126738,
   3.783958,
   5.055094,
   2.561565,
   2.937884,
   3.760461,
   4.477241,
   5.37765};
   gre = new TGraphErrors(20,simultaneous_fx1002,simultaneous_fy1002,simultaneous_fex1002,simultaneous_fey1002);
   gre->SetName("simultaneous");
   gre->SetTitle("");
   gre->SetFillStyle(1000);

   ci = TColor::GetColor("#009900");
   gre->SetLineColor(ci);

   ci = TColor::GetColor("#009900");
   gre->SetMarkerColor(ci);
   gre->SetMarkerStyle(24);
   gre->SetMarkerSize(1.3);
   
   TH1F *Graph_simultaneous1002 = new TH1F("Graph_simultaneous1002","",100,0,22.05);
   Graph_simultaneous1002->SetMinimum(-8.531898);
   Graph_simultaneous1002->SetMaximum(8.574993);
   Graph_simultaneous1002->SetDirectory(0);
   Graph_simultaneous1002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_simultaneous1002->SetLineColor(ci);
   Graph_simultaneous1002->GetXaxis()->SetLabelFont(42);
   Graph_simultaneous1002->GetXaxis()->SetTitleOffset(1);
   Graph_simultaneous1002->GetXaxis()->SetTitleFont(42);
   Graph_simultaneous1002->GetYaxis()->SetLabelFont(42);
   Graph_simultaneous1002->GetYaxis()->SetTitleFont(42);
   Graph_simultaneous1002->GetZaxis()->SetLabelFont(42);
   Graph_simultaneous1002->GetZaxis()->SetTitleOffset(1);
   Graph_simultaneous1002->GetZaxis()->SetTitleFont(42);
   gre->SetHistogram(Graph_simultaneous1002);
   
   multigraph->Add(gre,"");
   multigraph->Draw("ap");
   multigraph->GetXaxis()->SetLimits(0.095, 21.105);
   multigraph->GetXaxis()->SetTitle(" set");
   multigraph->GetXaxis()->SetLabelFont(42);
   multigraph->GetXaxis()->SetTitleOffset(1);
   multigraph->GetXaxis()->SetTitleFont(42);
   multigraph->GetYaxis()->SetTitle("ReHtilde");
   multigraph->GetYaxis()->SetLabelFont(42);
   multigraph->GetYaxis()->SetTitleFont(42);
   
   TLegend *leg = new TLegend(0.25,0.78,0.35,0.9,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("local_fit","local_fit","lpf");
   entry->SetFillStyle(1000);

   ci = TColor::GetColor("#0000ff");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(22);
   entry->SetMarkerSize(1.3);
   entry->SetTextFont(42);
   entry=leg->AddEntry("simultaneous","simultaneous","lpf");
   entry->SetFillStyle(1000);

   ci = TColor::GetColor("#009900");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#009900");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(24);
   entry->SetMarkerSize(1.3);
   entry->SetTextFont(42);
   leg->Draw();
   
   TPaveText *pt = new TPaveText(0.4359299,0.94,0.5640701,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("ReHtilde");
   pt->Draw();
   c4->Modified();
   c4->cd();
   c4->SetSelected(c4);
}
