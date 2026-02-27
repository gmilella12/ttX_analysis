import ROOT, sys, re
import ctypes
import numpy as np
import math
from array import array

# import pandas as pd

ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptDate(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFile(0)
ROOT.gStyle.SetOptTitle(0)

ROOT.gStyle.SetCanvasBorderMode(0)
ROOT.gStyle.SetCanvasColor(ROOT.kWhite)

ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetGridColor(ROOT.kBlack)
ROOT.gStyle.SetGridStyle(2)
ROOT.gStyle.SetGridWidth(1)

ROOT.gStyle.SetEndErrorSize(2)
#ROOT.gStyle.SetErrorX(0.)
ROOT.gStyle.SetMarkerStyle(20)

ROOT.gStyle.SetHatchesSpacing(0.9)
ROOT.gStyle.SetHatchesLineWidth(2)

ROOT.gStyle.SetTitleColor(1, "XYZ")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(32, "XYZ")
ROOT.gStyle.SetTitleXOffset(1.135)
ROOT.gStyle.SetTitleOffset(1.32, "YZ")

ROOT.gStyle.SetLabelColor(1, "XYZ")
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelSize(29, "XYZ")

ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetStripDecimals(True)
ROOT.gStyle.SetNdivisions(1005, "X")
ROOT.gStyle.SetNdivisions(506, "Y")

ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

ROOT.gStyle.SetPaperSize(8.0*1.35,6.7*1.35)
ROOT.TGaxis.SetMaxDigits(4)
ROOT.TGaxis.SetExponentOffset(-0.07, 0.01, "y")
ROOT.gStyle.SetLineScalePS(2)

# ROOT.gStyle.SetPalette(57)
# ROOT.gStyle.SetPalette(ROOT.kRainBow)
ROOT.gStyle.SetPaintTextFormat(".3f")


COLORS = []

def newColorRGB(red, green, blue):
    newColorRGB.colorindex += 1
    color = ROOT.TColor(newColorRGB.colorindex, red, green, blue)
    COLORS.append(color)
    return color

def HLS2RGB(hue, light, sat):
    r, g, b = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    ROOT.TColor.HLS2RGB(
        int(round(hue*255.)),
        int(round(light*255.)),
        int(round(sat*255.)),
        r,g,b
    )
    return r.value/255., g.value/255., b.value/255.
    
def newColorHLS(hue,light,sat):
    r,g,b = HLS2RGB(hue,light,sat)
    return newColorRGB(r,g,b)

def createYearVariations(h, l, s):
    # Variation factors for different years
    return {
        '2016': newColorHLS(max(h - 0.15, 0), max(l - 0.2, 0), min(s + 0.2, 1)),  # Darker and less saturated for 2016
        '2017': newColorHLS(min(h + 0.15, 1), min(l + 0.2, 1), max(s - 0.2, 0))   # Lighter and more saturated for 2017
    }

def RGB2HLS(r, g, b):
    h = ctypes.c_float()
    l = ctypes.c_float()
    s = ctypes.c_float()

    ROOT.TColor.RGB2HLS(r, g, b, ctypes.byref(h), ctypes.byref(l), ctypes.byref(s))

    return h.value, l.value, s.value

newColorRGB.colorindex = 301


def makeColorTable(reverse=False):
    colorList = [
        [0.,newColorHLS(0.6, 0.47,0.6)],
        [0.,newColorHLS(0.56, 0.65, 0.7)],
        [0.,newColorHLS(0.52, 1., 1.)],
    ]

    colorList = [
    [0.,newColorHLS(0.9, 0.5, 0.9)],
    [0.,newColorHLS(0.9, 0.88, 1.0)],
    [0.,newColorHLS(0.9, 0.95, 1.0)],
    ]
    
    if reverse:
        colorList = reversed(colorList)

    lumiMin = min(map(lambda x:x[1].GetLight(),colorList))
    lumiMax = max(map(lambda x:x[1].GetLight(),colorList))

    for color in colorList:
        if reverse:
            color[0] = ((lumiMax-color[1].GetLight())/(lumiMax-lumiMin))
        else:
            color[0] = ((color[1].GetLight()-lumiMin)/(lumiMax-lumiMin))

    stops = numpy.array(list(map(lambda x:x[0],colorList)))
    red   = numpy.array(list(map(lambda x:x[1].GetRed(),colorList)))
    green = numpy.array(list(map(lambda x:x[1].GetGreen(),colorList)))
    blue  = numpy.array(list(map(lambda x:x[1].GetBlue(),colorList)))

    start = ROOT.TColor.CreateGradientColorTable(stops.size, stops, red, green, blue, 200)
    ROOT.gStyle.SetNumberContours(200)
    return

rootObj = []

def makeCanvas(name="cv", width=800, height=670):
    ROOT.gStyle.SetPaperSize(width*0.0135, height*0.0135)
    cv = ROOT.TCanvas(name, "", width, height)
    rootObj.append(cv)
    return cv

def makePad(x1, y1, x2, y2, name="pad"):
    pad = ROOT.TPad(name, name, x1, y1, x2, y2)
    return pad

def makePaveText(x1, y1, x2, y2, name="cv"):
    pave = ROOT.TPaveText(x1, y1, x2, y2, "NDC") # NDC sets coordinates relative to the canvas
    pave.pave.SetBorderSize(0)
    pave.SetFillColor(0) # Transparent fill
    pave.SetTextFont(43)
    pave.SetTextSize(29)
    # pave.SetLabel(name)
    # pave.AddText("Histogram 1")
    # pave.AddText("Histogram 2")
    return pave

def makeLegend(x1, y1, x2, y2):
    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetBorderSize(0)
    legend.SetTextFont(43)
    legend.SetTextSize(29)
    legend.SetFillStyle(0)
    rootObj.append(legend)
    return legend

def makeCMSText(x1, y1, additionalText=None, dx=0.088, size=30):
    pTextCMS = ROOT.TPaveText(x1, y1, x1, y1, "NDC")
    pTextCMS.AddText("CMS")
    pTextCMS.SetTextFont(63)
    pTextCMS.SetTextSize(size)
    pTextCMS.SetTextAlign(13)
    pTextCMS.SetBorderSize(0)
    rootObj.append(pTextCMS)
    pTextCMS.Draw("Same")

    if additionalText:
        pTextAdd = ROOT.TPaveText(x1+dx, y1, x1+dx, y1, "NDC")
        pTextAdd.AddText(additionalText)
        pTextAdd.SetTextFont(53)
        pTextAdd.SetTextSize(size)
        pTextAdd.SetTextAlign(13)
        pTextAdd.SetBorderSize(0)
        rootObj.append(pTextAdd)
        pTextAdd.Draw("Same")
    return

def adjustFrame(frame, x_label='none', y_label='none'):
    frame.GetYaxis().SetTitle(y_label)
    # frame.GetXaxis().SetTitle('subleading jet AK8 pNet_{TvsQCD}')
    # frame.GetXaxis().SetTitle('minimum AK8 BDT score')
    frame.GetXaxis().SetTitle(x_label)
    # frame.GetYaxis().SetRangeUser(0., 1.1)
    return frame

def makeLumiText(x1, y1, lumi, year, size=30):
    pText = ROOT.TPaveText(x1, y1, x1, y1, "NDC")
    # pText.AddText( "7.8 fb#lower[-0.8]{#scale[0.7]{-1}} (13.6 TeV)")
    if str(year) == '2022' or str(year) == '2022EE' or year == 'all_years_Run3':
        pText.AddText(str(lumi) + " fb#lower[-0.8]{#scale[0.7]{-1}} (13.6 TeV)")
    else:
        pText.AddText( str(lumi) + " fb#lower[-0.8]{#scale[0.7]{-1}} (13 TeV)") #str(lumi) +
        # pText.AddText( " (13 TeV)") #str(lumi) +
    pText.SetTextFont(63)
    pText.SetTextSize(size)
    pText.SetTextAlign(13)
    pText.SetBorderSize(0)
    rootObj.append(pText)
    pText.Draw("Same")
    return
 
def makeText(x1,y1,x2, y2, text,size=30, font=43):
    pText = ROOT.TPaveText(x1, y1, x2, y2, "NBNDC")
    pText.AddText(text)
    pText.SetTextFont(font)
    pText.SetTextSize(size)
    pText.SetTextAlign(13)
    pText.SetBorderSize(0)
    pText.SetFillColorAlpha(ROOT.kWhite, 0.01)
    rootObj.append(pText)
    pText.Draw()
    return

def makeLine(x1, y1, x2, y2, size=30):
    line = ROOT.TLine(x1, y1, x2, y2)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(2)
    line.Draw('same')
    return

ptSymbol = "p#kern[-0.8]{ }#lower[0.3]{#scale[0.7]{T}}"
metSymbol = ptSymbol+"#kern[-2.3]{ }#lower[-0.8]{#scale[0.7]{miss}}"
metSymbol_lc = ptSymbol+"#kern[-2.3]{ }#lower[-0.8]{#scale[0.7]{miss,#kern[-0.5]{ }#mu-corr.}}}"
minDPhiSymbol = "#Delta#phi#lower[-0.05]{*}#kern[-1.9]{ }#lower[0.3]{#scale[0.7]{min}}"
htSymbol = "H#kern[-0.7]{ }#lower[0.3]{#scale[0.7]{T}}"
mhtSymbol = "H#kern[-0.7]{ }#lower[0.3]{#scale[0.7]{T}}#kern[-2.2]{ }#lower[-0.8]{#scale[0.7]{miss}}"
rSymbol = mhtSymbol+"#lower[0.05]{#scale[1.2]{/}}"+metSymbol
rSymbol_lc = mhtSymbol+"#lower[0.05]{#scale[1.2]{/}}"+metSymbol_lc
mzSymbol = "m#lower[0.3]{#scale[0.7]{#mu#mu}}"
gSymbol = "#tilde{g}"
qbarSymbol = "q#lower[-0.8]{#kern[-0.89]{#minus}}"
mgSymbol = "m#lower[0.2]{#scale[0.8]{#kern[-0.75]{ }"+gSymbol+"}}"
chiSymbol = "#tilde{#chi}#lower[-0.5]{#scale[0.65]{0}}#kern[-1.2]{#lower[0.6]{#scale[0.65]{1}}}"
mchiSymbol = "m#lower[0.2]{#scale[0.8]{"+chiSymbol+"}}"

def clamp(val, minimum=0, maximum=255):
    if val < minimum: return minimum
    if val > maximum:
        return maximum
    return int(val)

def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)
    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)

LUMINOSITY = {
    '2018': '59', '2017': '41.4',
    '2016preVFP': '19.5', '2016': '16.5',
    '2022': '7.8', '2022EE': '26.7',
    'all_years_Run2': '138',
    'all_years_Run3': '34.7', 
}
def additional_text(region, lepton_selection, is_data, year, miscellanea='', x1=0.16, x2=0.35, y1=0.85, y2=0.87):
    if is_data:
        makeCMSText(0.138, 0.965, additionalText="Preliminary") #Private Work
    else:
        makeCMSText(0.138, 0.95, additionalText="Simulation")

    if year == 'all_years_Run2':
        makeLumiText(0.63, 0.96, lumi=LUMINOSITY[str(year)], year=year)
    elif year == 'all_years_Run3':
        makeLumiText(0.595, 0.96, lumi=LUMINOSITY[str(year)], year=year)
    else:
        makeLumiText(0.75, 0.96, lumi=LUMINOSITY[str(year)], year=year)

    lepton_text = {'emu': "OS, e#mu", 'all_leptons': 'Leptons Merged'}
    lepton_text['no_lepton'] = ''
    if 'SR' in region:
        lepton_text['mumu'] = "#mu^{+}#mu^{-}, m_{#mu^{+}#mu^{-}}#notin [80, 101] GeV"
        lepton_text['ee'] = "e^{+}e^{-}, m_{e^{+}e^{-}}#notin [80, 101] GeV"
        lepton_text['ll'] = "l^{+}l^{-}, m_{l^{+}l^{-}}#notin [80, 101] GeV"
    elif 'noZrequirement' in region:
        lepton_text['mumu'] = "#mu^{+}#mu^{-}"
        lepton_text['ee'] = "e^{+}e^{-}"
        lepton_text['ll'] = "l^{+}l^{-}"
    elif 'single_muon' in region or 'single_muon_1T' in region:
        lepton_text['single_muon'] = '1 #mu, p_{T}(#mu+p_{T}^{miss})>250 GeV'
    else:
        lepton_text['mumu'] = "#mu^{+}#mu^{-}, m_{#mu^{+}#mu^{-}}#in [80, 101] GeV"
        lepton_text['ee'] = "e^{+}e^{-}, m_{e^{+}e^{-}}#in [80, 101] GeV"
        lepton_text['ll'] = "#font[12]{l}^{+}#font[12]{l}^{-}, m_{#font[12]{l}^{+}#font[12]{l}^{-}}#in [80, 101] GeV"

    makeText(x1, 0.85, x2, 0.87, lepton_text[lepton_selection], size=25)

    if region == 'single_muon':
        makeText(x1, 0.75, x2, 0.78, "#geq1b, #geq1J", size=25)
    if region == 'single_muon_1T':
        makeText(x1, 0.75, x2, 0.78, "#geq1b, #geq1T ", size=25)
        # makeText(x1, 0.72, x2, 0.75, "Top-tagged HOTVR ", size=25)

    # lepton_text['mumu'] = "#mu^{#pm}#mu^{#mp}"
    # lepton_text['ee'] = "e^{#pm}e^{#mp}"
    # lepton_text['ll'] = "e^{#pm}e^{#mp} + #mu^{#pm}#mu^{#mp}" #"l^{#pm}l^{#mp}"
    # lepton_text['emu'] ="e^{#pm}#mu^{#mp}"
    # if 'SR' in region and '0T' in region:
    #     makeText(x1, 0.82, x2, 0.85, f'{region.replace("SR", "TR")}', size=25)
    # else:
    #     if '1T' in region:
    #         region = region.replace("ex", "")
    #         region = region.replace("SR", "VR")
    #     region = region.replace("_noZrequirement", "")
    #     makeText(x1, 0.85, x2, 0.87, region, size=25)
    # makeText(x1, 0.77, x2, 0.8, lepton_text[lepton_selection], size=25)

    # makeText(x1, 0.78, x2, 0.81, "e^{#pm}e^{#mp} + #mu^{#pm}#mu^{#mp} ", size=25)

    return

def hex_to_root_color(hex_color):
    h = hex_color.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return ROOT.TColor.GetColor(*rgb)
HIST_COLORS = { 
    'dy': hex_to_root_color("#5790fc"), #newColorHLS(0.6, 0.7, 0.8).GetNumber(), 
    'DY-m-10_50': newColorHLS(0.6, 0.8, 0.9).GetNumber(),
    'DY-m-50': newColorHLS(0.6, 0.9, 0.8).GetNumber(),
    'VV': hex_to_root_color("#717581"),
    'qcd': hex_to_root_color("#e76300"), 
    'ttV': hex_to_root_color("#f89c20"), #newColorHLS(0.35, 0.7, 0.8).GetNumber(), 
    'ttX': hex_to_root_color("#f89c20"),
    'ST': hex_to_root_color("#e42536"), #ROOT.kRed+2, 
    '4t': hex_to_root_color("#7a21dd"), 
    # 'ttX': hex_to_root_color("#b9ac70"), 
    'tt': hex_to_root_color("#964a8b"), #newColorHLS(0.0, 0.7, 0.8).GetNumber(), 
    'ttH': hex_to_root_color("#9c9ca1"), #newColorHLS(0.12, 0.7, 0.8).GetNumber() 
    'tttX': hex_to_root_color("#a96b59"),
    'tZq': hex_to_root_color("#717581"),
    'multitop': hex_to_root_color("#7a21dd"), 
    'wjets': hex_to_root_color("#a96b59"),
}
HIST_LABELS = {
    'VV': 'VV',
    'tt': 'tt',
    'DY': 'DY',
    'ttX': 'ttX (X=V,H)',
    'ST': 'ST',
    'multitop': '3t+4t',
    'QCD': 'QCD',
    'wjets': 'W+j'
}

VARIABLES_BINNING = {
    'hotvr_invariant_mass_leading_subleading': array('d', [0., 600, 800, 1000, 1240, 1500, 2000, 5100]), # Run 2
    # 'hotvr_invariant_mass_leading_subleading': array('d', [0., 800, 1000, 1500, 2000, 5100]), # 600, 
    # 'hotvr_invariant_mass_leading_subleading': array('d', [0., 650, 800, 1100, 1500, 1900, 5100]), #Run 3
    # 'hotvr_invariant_mass_leading_subleading': array('d', list(np.arange(0., 5105., 100.))), # 600,
    'nhotvr': array('d', list(np.arange(0., 6))),
    'ntagged_hotvr': array('d', list(np.arange(0., 6))),
    'dilepton_invariant_mass_leading_subleading': array('d', list(np.arange(0., 1005., 1.))), #array('d', list(np.arange(0., 1005., 2.))),
    'eta': array('d', list(np.linspace(-2.5, 2.5, 101))),
    'phi': array('d', list(np.linspace(-3.5, 3.5, 101))),
    'met_and_muon_pt': array('d', list(np.arange(0., 3025., 10))),
    'nmuons': array('d', list(range(0, 5))),
    'nelectrons': array('d', list(range(0, 5))),
    'PV_npvsGood': array('d',list(range(0, 100))),
    'Nb': array('d', list(np.arange(0., 11.))),
    'Ntop': array('d', list(np.arange(0., 6.))),
    'MET_energy': array('d', list(np.concatenate([np.arange(0, 200, 20), np.arange(200, 300, 50), np.arange(300, 3026, 25) ]))),   
    'hotvr_MET_energy': array('d', list(np.concatenate([np.arange(0, 200, 20), np.arange(200, 400, 50), np.arange(400, 3026, 400) ]))),   
    'ht_ak4_and_hotvr': array('d', [150, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 900.0, 960.0, 1000.0, 1060.0, 1320.0, 2500., 5000.]),
}

for jet_type in ['leading', 'subleading']:
    VARIABLES_BINNING[f'hotvr_mass_{jet_type}'] = array('d', list(np.arange(0., 1510., 10)))#array('d', list(np.arange(0., 1510., 10)))
    VARIABLES_BINNING[f'hotvr_pt_{jet_type}'] = array('d', [150, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 900.0, 960.0, 1000.0, 1060.0, 1320.0])  #array('d', [150] + list(range(200, 900, 20)) + [900, 960, 1000, 1060, 1320])
    VARIABLES_BINNING[f'hotvr_eta_{jet_type}'] = array('d', list(np.arange(-2.5, 2.52, 0.3)))
    VARIABLES_BINNING[f'hotvr_phi_{jet_type}'] = array('d',list(np.arange(-3.5, 3.5, 0.3)))
    VARIABLES_BINNING[f'hotvr_tau3_over_tau2_{jet_type}'] = array('d', list(np.arange(0., 1.01, .02)))
    VARIABLES_BINNING[f'hotvr_scoreBDT_{jet_type}'] = array('d', list(np.arange(0., 1.05, .05)))
    VARIABLES_BINNING[f'hotvr_fractional_subjet_pt_{jet_type}'] = array('d', list(np.arange(0., 1.05, .05)))
    VARIABLES_BINNING[f'hotvr_min_pairwise_subjets_mass_{jet_type}'] = array('d', list(np.arange(0., 205., 10.)))
    VARIABLES_BINNING[f'hotvr_nsubjets_{jet_type}'] = array('d', list(np.arange(0., 5)))
    VARIABLES_BINNING[f'hotvr_pt_vs_mass_{jet_type}'] = array('d', list(np.arange(0., 5)))
    VARIABLES_BINNING[f'{jet_type}_pt'] = array('d', list(np.arange(0., 3025., 10))) 
    VARIABLES_BINNING[f'{jet_type}_phi'] = array('d',list(np.arange(-3.5, 3.5, 0.3)))
    VARIABLES_BINNING[f'{jet_type}_eta'] = array('d', list(np.arange(-2.5, 2.52, 0.3)))
    VARIABLES_BINNING[f'{jet_type}_pfRelIso04_all'] = array('d', list(np.arange(0., 0.2, 0.005)))
    VARIABLES_BINNING[f'ak4_outside_hotvr_pt_{jet_type}'] = array('d', np.concatenate([np.arange(30., 200., 10), np.arange(200., 500., 15), np.arange(500., 1000., 50), np.arange(1000., 2020., 1000)]))
    VARIABLES_BINNING[f'ak4_outside_hotvr_phi_{jet_type}'] = array('d',list(np.arange(-3.5, 3.5, 0.3)))
    VARIABLES_BINNING[f'ak4_outside_hotvr_eta_{jet_type}'] = array('d', list(np.arange(-2.5, 2.52, 0.3)))

def rebin_histogram_with_overflow(histo, bin_edges, new_name):
    new_histo = ROOT.TH1F(new_name, histo.GetTitle(), len(bin_edges) - 1, bin_edges)
    new_histo.SetEntries(histo.GetEntries())

    for i in range(1, histo.GetNbinsX() + 1):
        content = histo.GetBinContent(i)
        error = histo.GetBinError(i)
        bin_center = histo.GetBinCenter(i)
        
        new_bin = new_histo.FindBin(bin_center)
        new_histo.AddBinContent(new_bin, content)
        new_histo.SetBinError(new_bin, (new_histo.GetBinError(new_bin)**2 + error**2)**0.5)

    # underflow_content = histo.GetBinContent(0)
    # underflow_error = histo.GetBinError(0)

    # if underflow_content == 0.:
    #     underflow_content = 1e-8  # Small value to avoid log scale issues
    #     underflow_error = 1e-8

    # new_histo.AddBinContent(1, underflow_content)
    # new_histo.SetBinError(1, (new_histo.GetBinError(1)**2 + underflow_error**2)**0.5)

    # overflow_bin = histo.GetNbinsX() + 1
    # overflow_content = histo.GetBinContent(overflow_bin)
    # overflow_error = histo.GetBinError(overflow_bin)

    # if overflow_content == 0.:
    #     overflow_content = 1e-8  # Small value to avoid log scale issues
    #     overflow_error = 1e-8

    # new_histo.AddBinContent(new_histo.GetNbinsX(), overflow_content)
    # new_histo.SetBinError(new_histo.GetNbinsX(), (new_histo.GetBinError(new_histo.GetNbinsX())**2 + overflow_error**2)**0.5)

    return new_histo

def rebin_histogram_2d_with_overflow(histo, x_bin_edges, y_bin_edges, new_name):
    new_histo = ROOT.TH2F(
        new_name, 
        '', 
        len(x_bin_edges) - 1, 
        x_bin_edges, 
        len(y_bin_edges) - 1, 
        y_bin_edges
    )
    new_histo.SetEntries(histo.GetEntries())

    for ix in range(1, histo.GetNbinsX() + 1):
        for iy in range(1, histo.GetNbinsY() + 1):
            content = histo.GetBinContent(ix, iy)
            error = histo.GetBinError(ix, iy)
            x_bin_center = histo.GetXaxis().GetBinCenter(ix)
            y_bin_center = histo.GetYaxis().GetBinCenter(iy)

            new_bin = new_histo.FindBin(x_bin_center, y_bin_center)
            if content > 0. and new_bin > (new_histo.GetNbinsX() + 2) * (new_histo.GetNbinsY() + 2): 
                print(new_bin, content)
                print(x_bin_center, y_bin_center)
                # new_histo.SetBinContent(new_histo.GetNbinsX()+1, new_histo.GetNbinsY()+1, new_histo.GetBinContent(new_bin) + content)
                # new_histo.SetBinError(new_bin, math.sqrt(new_histo.GetBinError(new_bin)**2 + error**2))

            new_histo.SetBinContent(new_bin, new_histo.GetBinContent(new_bin) + content)
            new_histo.SetBinError(new_bin, math.sqrt(new_histo.GetBinError(new_bin)**2 + error**2))

    return new_histo

XSEC_UNC = {
    'tt':
    {   
        'Up': 1.05,
        'Down': 0.95  
    },
    'wjets':
    {
        'Up': 1.30,
        'Down': 0.70 
    },
}

# LEPTON_SELECTIONS = ['ee', 'emu', 'mumu']
SYSTEMATICS = [
    'nominal',

    'trigger', 
    'muonID', 
    'muonISO',
    'electronID', 
    'electronPt', 
    'PU', 
    'PUID',
    'bTaggingBC', 
    'bTaggingLight',
    'bTaggingBCCorrelated', 
    'bTaggingLightCorrelated',

    'hotvrJESTotal', 
    'hotvrJER', 
    'ISR', 
    'FSR',

    'MEfac', 
    'MEren',
    'MEenv',
    'PDF',
    'L1preFiring',

    'BDT',

    'xSecTTbar',
    'xSecWJets',
    'xSecTTTT',
    'xSecTTX',
]
UNCORRELATED_SYSTEMATICS = [
        # 'nominal'
        'trigger', 
        # 'muonISO',
        # 'muonID', 
        'bTaggingBC', 
        'bTaggingLight', 
        'hotvrJESTotal', 
        'hotvrJER', 
        'BDT',
        'L1preFiring',
        'PUID'
]

MASSES = [
    '500', '750', '1000', '1250', '1500', '1750', '2000', '2500', '3000', '4000'
]
WIDTHS = [
    '4', '10', '20', '50'
]
def process_sgn(input_root_file, directory, variable, additional_flag="", rebinning=False, signal=''):
    print('\nProcessing sgn...')
    histos_sgn = {}

    if not input_root_file.GetDirectory(directory):
        return histos_sgn

    for mass in MASSES:
        for width in WIDTHS: 
            histos_sgn[f"{mass}_{width}"] = None

            histo_name = f'TTZprimeToTT_M-{mass}_Width{width}_{variable}'
            if signal == 'tta':
                histo_name = f'tta_m{mass}_w{width}_{variable}'
            if signal == 'tth':
                histo_name = f'tth_m{mass}_w{width}_{variable}'
            if signal == 'TZPrime':
                histo_name = f'TZPrimetoTT_M{mass}_width{width}_{variable}'
            histo = input_root_file.Get(f'{directory}{histo_name}{additional_flag}')
            
            if 'TTZ' in histo_name:
                histo_name = histo_name.replace("TTZprimeToTT", "ttZprime")
            if 'TZPrime' == histo_name:
                histo_name = histo_name.replace("TZPrime", "tZprime")
                histo_name = re.sub(r'_w(\d+)', r'_W\1', histo_name)
                histo_name = re.sub(r'_M(\d+)', r'_M-\1', histo_name)
            if signal in ['tta', 'tth']:
                histo_name = re.sub(r'_m(\d+)', r'_M-\1', histo_name)
                histo_name = re.sub(r'_w(\d+)', r'_Width\1', histo_name)

            if not histo: 
                print(f"Histo {histo_name} not found!")
                continue
            else:
                if rebinning:
                    if isinstance(histo, ROOT.TH1F) or isinstance(histo, ROOT.TH1D):
                        histo = rebin_histogram_with_overflow(
                            histo, 
                            VARIABLES_BINNING[variable], 
                            histo_name
                        )
                    else:
                        if variable == 'eta_vs_phi':
                            varx = VARIABLES_BINNING['eta']
                            vary = VARIABLES_BINNING['phi']
                        if variable == 'Nb_outside_vs_Ntop':
                            varx = VARIABLES_BINNING['Ntop']
                            vary = VARIABLES_BINNING['Nb']
                        histo = rebin_histogram_2d_with_overflow(
                            histo, 
                            varx,
                            vary,
                            histo_name
                        )

                # histo.SetName(f"ttZprime_M-{mass}_Width{width}_{variable}")

                if width == '4':
                    print(
                        histo.GetName()+additional_flag,
                        histo.Integral(), 
                        histo.GetEntries()
                    )

                if histos_sgn[f'{mass}_{width}'] == None: 
                    histos_sgn[f'{mass}_{width}'] = histo
                else: 
                    histos_sgn[f'{mass}_{width}'].Add(histo)

                histo.SetDirectory(0)
                histos_sgn[f'{mass}_{width}'].SetDirectory(0)

    return histos_sgn

def process_data(input_root_file, directory, variable, additional_flag="", rebinning=False):
    print('\nProcessing data...')
    histo_data = None

    if not input_root_file.GetDirectory(directory):
        return histo_data

    input_root_file.cd(directory)
    current_dir = ROOT.gDirectory
    for histo_key in current_dir.GetListOfKeys():
        
        key_name = histo_key.GetName()
        pattern = re.compile(rf'_{re.escape(variable)}{re.escape(additional_flag)}$')
        if not pattern.search(key_name):
            continue
        if additional_flag == "":
            if re.search(r'_(ex1tag|[12]tag|less1tag)$', key_name):
                # print(key_name)
                continue

        if 'Double' in key_name or 'Muon' in key_name or 'Electron' in key_name: 
            #no need of separation of data samples as they are already separated in the input file 
            # if 'ee' in directory:
            #     histo_name = f'DoubleEG_{variable}'
            # elif 'mumu' in directory:
            #     histo_name = f'DoubleMuon_{variable}'
            # elif 'emu' in directory:
            #     histo_name = f'DoubleLepton_{variable}'

            histo = histo_key.ReadObj()

            histo.SetDirectory(0)
            print(histo.GetName(), histo.Integral())

            if histo_data is None:
                histo_data = histo.Clone()
                histo_data.SetDirectory(0)

                histo_name = f'data_{variable}'
                histo_data.SetName(histo_name)
            else: 
                histo_data.Add(histo)

    if histo_data != None:
        if rebinning: 
            if isinstance(histo_data, ROOT.TH1F) or isinstance(histo_data, ROOT.TH1D):
                histo_data = rebin_histogram_with_overflow(
                    histo_data, 
                    VARIABLES_BINNING[variable], 
                    histo_data.GetName(),
                )
            else:
                if variable == 'eta_vs_phi':
                    varx = VARIABLES_BINNING['eta']
                    vary = VARIABLES_BINNING['phi']
                if variable == 'Nb_outside_vs_Ntop':
                    varx = VARIABLES_BINNING['Ntop']
                    vary = VARIABLES_BINNING['Nb']
                histo_data = rebin_histogram_2d_with_overflow(
                    histo_data, 
                    varx,
                    vary,
                    histo_data.GetName(),
                )
            histo_data.SetDirectory(0)

    return histo_data

PROCESSES = [
    'tot_bkg', 
    'dy',
    'qcd',
    'tt',
    # 'ttV',
    # 'ttH', 
    'ttX',
    '4t',
    # 'tttX',
    'multitop',
    'ST',
    'VV',
    'wjets',
]
def process_bkg(input_root_file, directory, variable, additional_flag="", rebinning=False, year="", systematic=""):
    print('\nProcessing bkg...')
    histos_bkg = {}

    if not input_root_file.GetDirectory(directory):
        return histos_bkg

    input_root_file.cd(directory)
    current_dir = ROOT.gDirectory
    if not current_dir:
        print(f"Error: Could not access directory {directory}")
        return {}

    for histo_key in current_dir.GetListOfKeys():
        key_name = histo_key.GetName()
        
        pattern = re.compile(rf'_{re.escape(variable)}{re.escape(additional_flag)}$')
        if not pattern.search(key_name):
            continue
        if 'TTZ' in key_name:
            continue
        if 'tta' in key_name or 'tth' in key_name:
            continue
        if 'Double' in key_name or 'Muon' in key_name or 'Electron' in key_name:
            continue
        if 'dy_m-10_50' in key_name and '_lo' not in key_name: #NLO DY at low mass
            continue
        if year == '2022EE':
            if 'dy_m-50' in key_name:
                continue
        if '202' in year:
            if 'tW_dilepton' in key_name or 'tbarW_dilepton' in key_name:
                continue

        if additional_flag == "":
            if re.search(r'_(ex1tag|[12]tag|less1tag)$', key_name):
                # print(key_name)
                continue

        histo = histo_key.ReadObj()
        histo_name = histo.GetName()
        if histo.GetEntries() <= 1:
            continue

        print(histo_name, histo.Integral(), histo.GetEntries())

        process = ''
        if 'dy_' in histo_name: #no need of separation of data samples as they are already separated in the input file 
            process = 'DY'
            histo_new_name = f'dy_{variable}'

            if 'dy_m-10_50_hotvr' in histo_name: #NLO DY at low mass
                continue

        elif 'WW' in histo_name or 'ZZ' in histo_name or 'WZ' in histo_name:
            process = 'VV'
            histo_new_name = f'VV_{variable}'

        elif 'qcd' in histo_name:
            process = 'QCD'
            histo_new_name = f'qcd_{variable}'
        
        elif 'ttWJets' in histo_name or 'ttZJets' in histo_name or 'ttl' in histo_name or 'ttH' in histo_name:
            process = 'ttX'
            histo_new_name = f'ttX_{variable}'

        # elif 'ttH' in histo_name:
        #     process = 'ttH'
        #     histo_new_name = f'ttH_{variable}'

        elif 'tt_dilepton' in histo_name or 'tt_semilepton' in histo_name:
            process = 'tt'
            histo_new_name = f'tt_{variable}'

        elif ('tW_' in histo_name or 'tbarW' in histo_name or 'ST_' in histo_name) and 'ttt' not in histo_name: 
            process = 'ST'
            histo_new_name = f'ST_{variable}'

        elif 'tttt' in histo_name or 'tttW' in histo_name:
            process = 'multitop'
            histo_new_name = f'multitop_{variable}'

        # elif 'ttHH' in histo_name or 'ttWH' in histo_name or 'ttWW' in histo_name or 'ttWZ' in histo_name or 'ttZH' in histo_name or 'ttZZ' in histo_name or 'tttJ' in histo_name or 'tttW' in histo_name:
        #     process = 'tttX'
        #     histo_new_name = f'tttX_{variable}'

        elif 'wjets' in histo_name or 'w_to' in histo_name:
            process = 'wjets'
            histo_new_name = f'wjets_{variable}'

        elif 'tZq' in histo_name:
            process = 'tZq'
            histo_new_name = f'tZq_{variable}'

        else: 
            continue

        # xsec norm uncertainty
        if systematic == 'xSecTTbarUp' and ('tt_dilepton' in histo_name or 'tt_semilepton' in histo_name):
            histo.Scale(XSEC_UNC['tt']['Up'])
        if systematic == 'xSecTTbarDown' and ('tt_dilepton' in histo_name or 'tt_semilepton' in histo_name):
            histo.Scale(XSEC_UNC['tt']['Down'])
        if systematic == 'xSecWJetsUp' and 'wjets_' in histo_name:
            histo.Scale(XSEC_UNC['wjets']['Up'])
        if systematic == 'xSecWJetsDown' and 'wjets_' in histo_name:
            histo.Scale(XSEC_UNC['wjets']['Down'])

        if process not in histos_bkg:
            histos_bkg[process] = histo.Clone()
            histos_bkg[process].SetDirectory(0)
            histos_bkg[process].SetName(histo_new_name)
            histos_bkg[process].SetEntries(histo.GetEntries())
        else:
            histos_bkg[process].Add(histo)
            new_entries = histos_bkg[process].GetEntries() + histo.GetEntries()
            histos_bkg[process].SetEntries(new_entries)

        if process != "tZq":
            if 'tot_bkg' not in histos_bkg:
                histos_bkg['tot_bkg'] = histo.Clone()
                histos_bkg['tot_bkg'].SetDirectory(0)
                histos_bkg['tot_bkg'].SetName(f'tot_bkg_{variable}')
                histos_bkg['tot_bkg'].SetEntries(histo.GetEntries())
            else:
                histos_bkg['tot_bkg'].Add(histo)
                new_total_entries = histos_bkg['tot_bkg'].GetEntries() + histo.GetEntries()
                histos_bkg['tot_bkg'].SetEntries(new_total_entries)


    for process in histos_bkg.keys():
        if histos_bkg[process] != None:
            if rebinning:
                if isinstance(histos_bkg[process], ROOT.TH1F) or isinstance(histos_bkg[process], ROOT.TH1D):
                    histos_bkg[process] = rebin_histogram_with_overflow(
                        histos_bkg[process], 
                        VARIABLES_BINNING[variable], 
                        histos_bkg[process].GetName(),
                    )
                else:
                    if variable == 'eta_vs_phi':
                        varx = VARIABLES_BINNING['eta']
                        vary = VARIABLES_BINNING['phi']
                    if variable == 'Nb_outside_vs_Ntop':
                        varx = VARIABLES_BINNING['Ntop']
                        vary = VARIABLES_BINNING['Nb']
                    histos_bkg[process] = rebin_histogram_2d_with_overflow(
                        histos_bkg[process], 
                        varx,
                        vary,
                        histos_bkg[process].GetName(),
                    )
                    
                histos_bkg[process].SetDirectory(0)
             
        print(process, histos_bkg[process].Integral(), histos_bkg[process].GetEntries())

    return histos_bkg

def process_bkg_jet_flavors(input_root_file, directory, variable, additional_flag="", rebinning=False, year='', flavor='', systematic=""):
    print(f'\nProcessing bkg (per flavor [{flavor}])...')
    histos_bkg = {}

    if not input_root_file.GetDirectory(directory):
        return histos_bkg

    input_root_file.cd(directory)
    current_dir = ROOT.gDirectory
    for histo_key in current_dir.GetListOfKeys():
        key_name = histo_key.GetName()
        
        if f'{variable}{additional_flag}' not in key_name:
            continue
        
        if additional_flag == "":
            if re.search(r'_(ex1tag|[12]tag|less1tag)$', key_name):
                # print(key_name)
                continue
        if 'Double' in key_name or 'Muon' in key_name or 'Electron' in key_name:
            continue
        if 'dy_m-10_50_hotvr' in key_name and '_lo' not in key_name: #NLO DY at low mass
                continue
        if year == '2022EE':
            if 'dy_m-50' in key_name:
                continue
        if '202' in year:
            if 'tW_dilepton'in key_name or 'tbarW_dilepton' in key_name:
                continue

        process = ''
        # if flavor in key_name: 
        if key_name.endswith(flavor):
            if flavor == 'b_quark':
                if '_no_' in key_name or '_one_' in key_name:
                    continue
            if flavor == 'hadronic_t':
                if '_quark' in key_name or 'qcd' in key_name:
                    continue

            histo = histo_key.ReadObj()
            histo_name = histo.GetName()
            histo.SetDirectory(0)
            print(histo.GetName(), histo.Integral(), histo.GetEntries())

            if 'dy_' in histo_name: #no need of separation of data samples as they are already separated in the input file 
                process = 'DY'
                histo_new_name = f'dy_{variable}'

                if 'dy_m-10_50_hotvr' in histo_name: #NLO DY at low mass
                    continue

            elif 'WW' in histo_name or 'ZZ' in histo_name or 'WZ' in histo_name:
                process = 'VV'
                histo_new_name = f'VV_{variable}'

            elif 'qcd' in histo_name:
                process = 'QCD'
                histo_new_name = f'qcd_{variable}'
            
            elif 'ttWJets' in histo_name or 'ttZJets' in histo_name or 'ttl' in histo_name or 'ttH' in histo_name:
                process = 'ttX'
                histo_new_name = f'ttX_{variable}'

            # elif 'ttH' in histo_name:
            #     process = 'ttH'
            #     histo_new_name = f'ttH_{variable}'

            elif 'tt_dilepton' in histo_name or 'tt_semilepton' in histo_name:
                process = 'tt'
                histo_new_name = f'tt_{variable}'

            elif ('tW_' in histo_name or 'tbarW' in histo_name or 'ST_' in histo_name) and 'ttt' not in histo_name: 
                process = 'ST'
                histo_new_name = f'ST_{variable}'

            elif 'tttt' in histo_name or 'tttW' in histo_name:
                process = 'multitop'
                histo_new_name = f'multitop_{variable}'

            # elif 'ttHH' in histo_name or 'ttWH' in histo_name or 'ttWW' in histo_name or 'ttWZ' in histo_name or 'ttZH' in histo_name or 'ttZZ' in histo_name or 'tttJ' in histo_name or 'tttW' in histo_name:
            #     process = 'tttX'
            #     histo_new_name = f'tttX_{variable}'

            elif 'wjets' in histo_name or 'w_to' in histo_name:
                process = 'wjets'
                histo_new_name = f'wjets_{variable}'

            elif 'tZq' in histo_name:
                process = 'tZq'
                histo_new_name = f'tZq_{variable}'

            else: 
                continue

            if f'{process}_{flavor}' not in histos_bkg:
                histos_bkg[f'{process}_{flavor}'] = histo.Clone()
                histos_bkg[f'{process}_{flavor}'].SetDirectory(0)
                histos_bkg[f'{process}_{flavor}'].SetName(histo_new_name.replace(variable, f'{flavor}_{variable}'))
                # histos_bkg[f'{process}_{flavor}'].SetEntries(histo.GetEntries())
            else:
                histos_bkg[f'{process}_{flavor}'].Add(histo)
                # new_entries = histos_bkg[f'{process}_{flavor}'].GetEntries() + histo.GetEntries()
                # histos_bkg[f'{process}_{flavor}'].SetEntries(new_entries)

            if process != "tZq":
                if f'tot_bkg_{flavor}' not in histos_bkg:
                    histos_bkg[f'tot_bkg_{flavor}'] = histo.Clone()
                    histos_bkg[f'tot_bkg_{flavor}'].SetDirectory(0)
                    histos_bkg[f'tot_bkg_{flavor}'].SetName(f'tot_bkg_{flavor}_{variable}')
                    # histos_bkg[f'tot_bkg_{flavor}'].SetEntries(histo.GetEntries())
                else:
                    histos_bkg[f'tot_bkg_{flavor}'].Add(histo)
                    # new_total_entries = histos_bkg[f'tot_bkg_{flavor}'].GetEntries() + histo.GetEntries()
                    # histos_bkg[f'tot_bkg_{flavor}'].SetEntries(new_total_entries)

    for bkg_key in histos_bkg.keys():
        if histos_bkg[bkg_key] != None:
            if rebinning:
                if isinstance(histos_bkg[bkg_key], ROOT.TH1F) or isinstance(histos_bkg[bkg_key], ROOT.TH1D):
                    histos_bkg[bkg_key] = rebin_histogram_with_overflow(
                        histos_bkg[bkg_key], 
                        VARIABLES_BINNING[variable], 
                        histos_bkg[bkg_key].GetName(),
                    )
                else:
                    if variable == 'eta_vs_phi':
                        varx = VARIABLES_BINNING['eta']
                        vary = VARIABLES_BINNING['phi']
                    if variable == 'Nb_outside_vs_Ntop':
                        varx = VARIABLES_BINNING['Ntop']
                        vary = VARIABLES_BINNING['Nb']
                    histos_bkg[bkg_key] = rebin_histogram_2d_with_overflow(
                        histos_bkg[bkg_key], 
                        varx,
                        vary,
                        histos_bkg[bkg_key].GetName(),
                    )
                    
                histos_bkg[bkg_key].SetDirectory(0)
        print(bkg_key, histos_bkg[bkg_key].Integral(), histos_bkg[bkg_key].GetEntries())

    return histos_bkg

JET_COMPOSITIONS = [
    # 'pure_qcd',
    # 'hadronic_t',
    # 'hadronic_w',
    # 'b_from_top',
    # 'b_not_from_top',
    # 'q_from_w_plus_b',
    # 'q_from_w',
    # 'others',
    # 'hadronic_w_not_from_t',
    # 'q_from_w_not_from_top'
    't-matched',
    'W-matched',
    'non-matched'
]
def process_bkg_jet_composition(input_root_file, directory, variable, additional_flag="", rebinning=False, year="", systematic=''):
    histos_bkg = {}

    input_root_file.cd(directory)
    current_dir = ROOT.gDirectory
    if not current_dir:
        print(f"Error: Could not access directory {directory}")
        return {}

    for histo_key in current_dir.GetListOfKeys():
        key_name = histo_key.GetName()

        if 'TTZ' in key_name:
            continue
        if 'Double' in key_name or 'Muon' in key_name or 'Electron' in key_name:
            continue
        if 'dy_m-10_50_hotvr' in key_name and '_lo' not in key_name: #NLO DY at low mass
            continue
        if year == '2022EE':
            if 'dy_m-50' in key_name:
                continue
        if '202' in year:
            if 'tW_dilepton'in key_name or 'tbarW_dilepton' in key_name:
                continue

        if f'{variable.replace("hotvr", "")}{additional_flag}' not in key_name:
            # if 'mass' in variable:
            #     print(f'{variable.replace("hotvr", "")}{additional_flag}', variable.replace("hotvr", "").startswith('_mass'))
            #     sys.exit()
            continue
        
        if variable.replace("hotvr", "").startswith('_mass') and 'subjets' in key_name:
            continue
        
        if additional_flag == "":
            if re.search(r'_(ex1tag|[12]tag|less1tag)$', key_name):
                # print(key_name)
                continue

        histo = histo_key.ReadObj()
        histo_name = histo.GetName()
        print(histo_name, histo.Integral(), histo.GetEntries())

        flavor = ''

        if 'hadronic_t' in histo_name:
            # flavor = 'hadronic_t'
            # histo_new_name = f'hadronic_t_{variable}'
            flavor = 't-matched'
            histo_new_name = f't-matched_{variable}'

        elif ('hadronic_w' in histo_name and '_not_from_t' not in histo_name) or 'q_from_w_plus_b' in histo_name or ('q_from_w' in histo_name and '_not_from_t' not in histo_name) or 'hadronic_w_not_from_t' in histo_name or 'q_from_w_not_from_top' in histo_name:
            flavor = 'W-matched'
            histo_new_name = f'W-matched_{variable}'

        else:
            flavor = 'non-matched'
            histo_new_name = f'non-matched_{variable}'

        # elif 'hadronic_w' in histo_name and '_not_from_t' not in histo_name:
        #     flavor = 'hadronic_w'
        #     histo_new_name = f'hadronic_w_{variable}'

        # elif 'b_from_top' in histo_name:
        #     flavor = 'b_from_top'
        #     histo_new_name = f'b_from_top_{variable}'
        
        # elif 'b_not_from_top' in histo_name:
        #     flavor = 'b_not_from_top'
        #     histo_new_name = f'b_not_from_top_{variable}'

        # elif 'q_from_w_plus_b' in histo_name:
        #     flavor = 'q_from_w_plus_b'
        #     histo_new_name = f'q_from_w_plus_b_{variable}'

        # elif 'q_from_w' in histo_name and '_not_from_t' not in histo_name:
        #     flavor = 'q_from_w'
        #     histo_new_name = f'q_from_w_{variable}'

        # elif 'others' in histo_name or 'non_covered' in histo_name:
        #     flavor = 'others'
        #     histo_new_name = f'others_{variable}'

        # elif 'pure_qcd' in histo_name:
        #     flavor = 'pure_qcd'
        #     histo_new_name = f'pure_qcd_{variable}'

        # elif 'hadronic_w_not_from_t' in histo_name:
        #     flavor = 'hadronic_w_not_from_t'
        #     histo_new_name = f'hadronic_w_not_from_t_{variable}'

        # elif 'q_from_w_not_from_top' in histo_name:
        #     flavor = 'q_from_w_not_from_top'
        #     histo_new_name = f'q_from_w_not_from_top_{variable}'

        # else: 
        #     continue

        # xsec norm uncertainty
        if systematic == 'xSecTTbarUp' and ('tt_dilepton' in histo_name or 'tt_semilepton' in histo_name):
            histo.Scale(XSEC_UNC['tt']['Up'])
        if systematic == 'xSecTTbarDown' and ('tt_dilepton' in histo_name or 'tt_semilepton' in histo_name):
            histo.Scale(XSEC_UNC['tt']['Down'])
        if systematic == 'xSecWJetsUp' and 'wjets_' in histo_name:
            histo.Scale(XSEC_UNC['wjets']['Up'])
        if systematic == 'xSecWJetsDown' and 'wjets_' in histo_name:
            histo.Scale(XSEC_UNC['wjets']['Down'])

        if flavor not in histos_bkg.keys():
            histos_bkg[flavor] = histo.Clone()
            histo.SetDirectory(0)
            histos_bkg[flavor].SetDirectory(0)
            histos_bkg[flavor].SetName(histo_new_name)
        else:
            histos_bkg[flavor].Add(histo)

        if 'tot_bkg' not in histos_bkg:
            histos_bkg['tot_bkg'] = histo.Clone()
            histos_bkg['tot_bkg'].Reset()
            histos_bkg['tot_bkg'].SetDirectory(0)
            histos_bkg['tot_bkg'].SetName(f'tot_bkg_{variable}')
        else:
            histos_bkg['tot_bkg'].Add(histo)

    for flavor in histos_bkg.keys():
        if histos_bkg[flavor] != None:
            if rebinning: 
                histos_bkg[flavor] = rebin_histogram_with_overflow(
                    histos_bkg[flavor], 
                    VARIABLES_BINNING[variable], 
                    histos_bkg[flavor].GetName(),
                )
                histos_bkg[flavor].SetDirectory(0)
        print(flavor, histos_bkg[flavor].Integral())

    return histos_bkg

def process_with_additional_flag(histo_name, additional_flag):
    """
    Determines whether to process a histogram based on additional flags.
    Parameters:
        histo_name (str): The histogram name to check.
        additional_flag (list): A list defining inclusion, exclusion, and conditions.
                                Format: [include, condition, exclude, optional_condition]
                                - include: A string or list of strings to match.
                                - condition: 'in' or 'not_in' for inclusion/exclusion.
                                - exclude: Optional string or list of strings to exclude.
                                - optional_condition: Additional string or list to require.
    Returns:
        bool: True if the histogram should be processed, False otherwise.
    """
    include_flags = additional_flag[0]
    condition = additional_flag[1]
    exclude = additional_flag[2] if len(additional_flag) > 2 else None
    optional_condition = additional_flag[3] if len(additional_flag) > 3 else None

    # Check inclusion
    if condition == 'in':
        if isinstance(include_flags, list):
            if not any(flag in histo_name for flag in include_flags):
                return False
        elif include_flags not in histo_name:
            return False
    elif condition == 'not_in':
        if isinstance(include_flags, list):
            if any(flag in histo_name for flag in include_flags):
                return False
        elif include_flags in histo_name:
            return False
    else:
        raise ValueError(f"Invalid condition: {condition}. Must be 'in' or 'not_in'.")

    # Check exclusion
    if exclude:
        if isinstance(exclude, list):
            if any(flag in histo_name for flag in exclude):
                return False
        elif exclude in histo_name:
            return False

    # Enforce optional condition (e.g., 'onlyHadrTop')
    if optional_condition:
        if isinstance(optional_condition, list):
            if not all(flag in histo_name for flag in optional_condition):
                return False
        elif optional_condition not in histo_name:
            return False

    return True

def process_histogram_data(histo, histo_name, event_selection, lepton_selection, all_data, **kwargs):
    if 'Muon' not in histo_name and 'EG' not in histo_name and 'Lepton' not in histo_name:
        return

    if 'additional_flag' in kwargs.keys():
        additional_flag = kwargs['additional_flag']
        if not process_with_additional_flag(histo_name, additional_flag):
            return

    if 'variable' in kwargs.keys():
    # if kwargs['variable']+'_' not in histo_name: return
        if kwargs['variable'] not in histo_name: return

    print(histo_name)

    if all_data[lepton_selection] == None:
        all_data[lepton_selection] = histo.Clone('tot_data_lep_{}'.format(lepton_selection))
        ROOT.SetOwnership(all_data[lepton_selection], False)
    else: 
        all_data[lepton_selection].Add(histo)

    return

def process_histogram_bkg(histo, histo_name, systematic_variation, event_selection, lepton_selection, all_bkg, all_bkg_per_process, **kwargs):
    """
    Process a single histogram and update the all_bkg and all_bkg_per_process dictionaries.
    """
    if 'Double' in histo_name or 'Muon' in histo_name or 'Zprime' in histo_name:
        return
    if 'total' in histo_name: return
    if 'ttVJ_event' in histo_name or 'tt_event' in histo_name or 'ttH_event' in histo_name or 'dy_event' in histo_name: 
        return
    if 'dy_to_ll_' in histo_name: 
        return

    if 'additional_flag' in kwargs.keys():
        additional_flag = kwargs['additional_flag']  # Use the entire additional_flag structure
        if not process_with_additional_flag(histo_name, additional_flag):  # Only pass two arguments
            return

    if 'variable' in kwargs.keys():
        if '_'+kwargs['variable'] not in histo_name: return
        # if kwargs['variable'] not in histo_name: return

    # if event_scudd c election not in histo_name: return

    # if 'reconstructed' in histo_name: return
    # if 'dy_m-5' in histo_name and 'nlo' not in histo_name: return
    # if 'dy_m-1' in histo_name: return
    # if 'dy_m-5' in histo_name: return
    # if 'dy_ht' in histo_name: return

    if 'year' in kwargs.keys():
        if 'dy_to' in histo_name and kwargs['year'] == '2018': return
        if 'dy_m-50_dilepton' in histo_name and kwargs['year'] == '2022EE': return
    if 'nlo' in histo_name: return
    
    # if 'tttJ' in histo_name or 'tttW' in histo_name:
    #     return
    if 'tZq' in histo_name:
        return
    # if 'ZZ' in histo_name or 'WZ' in histo_name or 'WW' in histo_name:
    #     return
    # if 'tttt' in histo_name: 
    #     return

    # if "qcd" not in histo_name: return
    if histo.Integral() < 0 and histo.GetEntries() == 1:
        return
    if histo.GetEntries() == 1:
        return
    
    if 'region' in kwargs.keys():
        region = kwargs['region']
        # if 'SR' in region and 'm-10_50' in histo_name: 
        #     return
        if 'SR' in region and 'm-10_50' in histo_name: # and '_lo_' not in histo_name: 
            return

    print(histo_name, histo.Integral())

    if all_bkg[systematic_variation][lepton_selection] == None:
        all_bkg[systematic_variation][lepton_selection] = histo.Clone('tot_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
        all_bkg[systematic_variation][lepton_selection].SetDirectory(0)
        ROOT.SetOwnership(all_bkg[systematic_variation][lepton_selection], False)
    else: 
        all_bkg[systematic_variation][lepton_selection].Add(histo)

    if 'dy_' in histo_name:
        if 'DY' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['DY'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['DY'] = histo.Clone('dy_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['DY'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['DY'], False)

    # elif 'dy_m-50' in histo_name:
    #     if 'DY-m-50' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
    #         all_bkg_per_process[systematic_variation][lepton_selection]['DY-m-50'].Add(histo)
    #     else: 
    #         all_bkg_per_process[systematic_variation][lepton_selection]['DY-m-50'] = histo.Clone('dym50_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
    
    # elif 'dy_m-10' in histo_name:
    #     if 'DY-m-10_50' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
    #         all_bkg_per_process[systematic_variation][lepton_selection]['DY-m-10_50'].Add(histo)
    #     else: all_bkg_per_process[systematic_variation][lepton_selection]['DY-m-10_50'] = histo.Clone('dym10_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

    elif 'WZ' in histo_name or 'ZZ' in histo_name or 'WW' in histo_name:
        if 'VV' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['VV'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['VV'] = histo.Clone('VV_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['VV'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['VV'], False)

    elif 'qcd' in histo_name:
        if 'QCD' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['QCD'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['QCD'] = histo.Clone('qcd_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['QCD'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['QCD'], False)

    elif 'ttWJets' in histo_name or 'ttZJets' in histo_name or 'ttl' in histo_name:
        if 'ttVJ' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['ttVJ'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['ttVJ'] = histo.Clone('ttVJ_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['ttVJ'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['ttVJ'], False)

    elif 'tt_dilepton' in histo_name or 'tt_semilepton' in histo_name:
        if 'tt' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['tt'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['tt'] = histo.Clone('tt_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['tt'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['tt'], False)

    elif 'tW' in histo_name or 'tbarW' in histo_name or 'ST_' in histo_name:
        if 'tW' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['tW'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['tW'] = histo.Clone('tW_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['tW'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['tW'], False)

    elif 'tttt' in histo_name:
        if 'tttt' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['tttt'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['tttt'] = histo.Clone('tttt_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['tttt'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['tttt'], False)

    elif 'ttHH' in histo_name or 'ttWH' in histo_name or 'ttWW' in histo_name or 'ttWZ' in histo_name or 'ttZH' in histo_name or 'ttZZ' in histo_name or 'tttJ' in histo_name or 'tttW' in histo_name:
        if 'tttX' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['tttX'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['tttX'] = histo.Clone('tttX_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['tttX'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['tttX'], False)

    elif 'tZq' in histo_name:
        if 'tZq' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['tZq'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['tZq'] = histo.Clone('tZq_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['tZq'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['tZq'], False)

    elif 'ttH' in histo_name:
        if 'ttH' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
            all_bkg_per_process[systematic_variation][lepton_selection]['ttH'].Add(histo)
        else: 
            all_bkg_per_process[systematic_variation][lepton_selection]['ttH'] = histo.Clone('ttH_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
            all_bkg_per_process[systematic_variation][lepton_selection]['ttH'].SetDirectory(0)
            ROOT.SetOwnership(all_bkg_per_process[systematic_variation][lepton_selection]['ttH'], False)

    del histo

    return

def process_histogram_bkg_jet_composition(histo, histo_name, systematic_variation, event_selection, lepton_selection, all_bkg, all_bkg_per_process, **kwargs):
    """
    Process a single histogram and update the all_bkg and all_bkg_per_process dictionaries.
    """
    if 'Double' in histo_name or 'Muon' in histo_name or 'TTZ' in histo_name:
        return
    if 'total' in histo_name: return
    if 'ttVJ_event' in histo_name or 'tt_event' in histo_name or 'ttH_event' in histo_name or 'dy_event' in histo_name: 
        return
    # if 'tttt' in histo_name: return

    if 'additional_flag' in kwargs.keys():
        additional_flag = kwargs['additional_flag']  # Use the entire additional_flag structure
        if not process_with_additional_flag(histo_name, additional_flag):  # Only pass two arguments
            return

    if 'variable' in kwargs.keys():
        if '_'+kwargs['variable'] not in histo_name: return
    
    print(histo_name)

    # if event_selection not in histo_name: return
    if 'year' in kwargs.keys():
        if 'dy_to' in histo_name and kwargs['year'] == '2018': return
    if 'nlo' in histo_name: return

    if systematic_variation == 'nominal' and 'hotvr_'+kwargs['variable'] in histo_name: return
    if all_bkg[systematic_variation][lepton_selection] == None:
        all_bkg[systematic_variation][lepton_selection] = histo.Clone('tot_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
    else: 
        all_bkg[systematic_variation][lepton_selection].Add(histo)

    if systematic_variation == 'nominal':
        if 'hadronic_w' in histo_name and '_not_from_t' not in histo_name:
            if 'hadronic_w' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_w'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_w'] = histo.Clone('hadronic_w_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'b_from_top' in histo_name:
            if 'b_from_top' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['b_from_top'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['b_from_top'] = histo.Clone('b_from_top_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'b_not_from_top' in histo_name:
            if 'b_not_from_top' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['b_not_from_top'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['b_not_from_top'] = histo.Clone('b_not_from_top_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'q_from_w_plus_b' in histo_name:
            if 'q_from_w_plus_b' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w_plus_b'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w_plus_b'] = histo.Clone('q_from_w_plus_b_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'q_from_w' in histo_name and '_not_from_t' not in histo_name:
            if 'q_from_w' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w'] = histo.Clone('q_from_w_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'others' in histo_name or 'non_covered' in histo_name:
            if 'others' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['others'].Add(histo)
            else:
                all_bkg_per_process[systematic_variation][lepton_selection]['others'] = histo.Clone('others_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection)) 

        elif 'hadronic_t' in histo_name:
            if 'hadronic_t' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_t'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_t'] = histo.Clone('hadronic_t_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))
        
        elif 'pure_qcd' in histo_name:
            if 'pure_qcd' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['pure_qcd'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['pure_qcd'] = histo.Clone('pure_qcd_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection)) 

        elif 'hadronic_w_not_from_t' in histo_name:
            if 'hadronic_w_not_from_t' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_w_not_from_t'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['hadronic_w_not_from_t'] = histo.Clone('hadronic_w_not_from_t_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

        elif 'q_from_w_not_from_top' in histo_name:
            if 'q_from_w_not_from_top' in all_bkg_per_process[systematic_variation][lepton_selection].keys():
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w_not_from_top'].Add(histo)
            else: 
                all_bkg_per_process[systematic_variation][lepton_selection]['q_from_w_not_from_top'] = histo.Clone('q_from_w_not_from_top_bkg_sys_{}_lep_{}'.format(systematic_variation, lepton_selection))

    return

def process_histogram_sgn(histo, histo_name, systematic_variation, event_selection, lepton_selection, all_sgn, **kwargs):
    if 'Zprime' not in histo_name:
        return
    
    if kwargs.get('is_sgn_single_top', False):
        if 'TTZ' in histo_name.split("_", 2)[0]:
            return
    elif 'TZprimeToTT' == histo_name.split("_", 2)[0]:
        return

    if 'additional_flag' in kwargs.keys():
        additional_flag = kwargs['additional_flag']
        if not process_with_additional_flag(histo_name, additional_flag):
            return

    if 'variable' in kwargs.keys():
        if '_'+kwargs['variable'] not in histo_name: return

    # if event_selection not in histo_name: return
    # if 'reconstructed' in histo_name: return

    mass = ''
    pattern = r"M-(\d+)_Width(\d+)"
    match = re.search(pattern, histo_name)
    if match: 
        mass = match.group(1)
        width = match.group(2)
    else: 
        return

    print(histo_name)

    all_sgn[systematic_variation][lepton_selection][f'{mass}_{width}'] = histo.Clone(f'tot_sgn_{mass}_{width}_{lepton_selection}'.format())
    ROOT.SetOwnership(all_sgn[systematic_variation][lepton_selection][f'{mass}_{width}'], False)

    return

def create_systematic_variations(systematics):
    """
    Include 'nominal' as a systematic variation to simplify looping logic.
    """
    variations = {}
    for syst in systematics:
        if syst == 'nominal':
            variations[syst] = [None]  # No variation for nominal
        else:
            variations[syst] = ['Up', 'Down']
    return variations

def merging_histo_sgn(root_file, event_selection, lepton_selections, **kwargs):
    all_sgn = {}
    systematic_variations = create_systematic_variations(SYSTEMATICS)

    for systematic, variations in systematic_variations.items():
        print(systematic, variations)
        for variation in variations:
            variation_suffix = '' if variation is None else variation
            key = systematic if variation is None else "{}{}".format(systematic, variation_suffix)
            
            all_sgn[key] = {}

            for lepton_selection in lepton_selections:
                print(lepton_selection)
                directory = "{}/{}".format(systematic, lepton_selection) if variation is None else "{}{}/{}".format(
                    systematic, variation_suffix, lepton_selection
                )
                if root_file.GetDirectory(directory):
                    root_file.cd(directory)
                    current_dir = ROOT.gDirectory
                else:
                    print(f"Warning: Directory {directory} does not exist.")
                    continue

                all_sgn[key][lepton_selection] = {}

                # print(current_dir.GetListOfKeys())
                print('\nProcessing sgn...')
                for histo_key in current_dir.GetListOfKeys():
                    histo = histo_key.ReadObj()
                    process_histogram_sgn(histo, histo.GetName(), key, event_selection, lepton_selection, all_sgn, **kwargs)

    return all_sgn

def merging_histo_bkg(root_file, event_selection, lepton_selections, **kwargs):
    all_bkg, all_bkg_per_process = {}, {}
    systematic_variations = create_systematic_variations(SYSTEMATICS)

    print('\nProcessing bkgs...')
    for systematic, variations in systematic_variations.items():
        print('\nSys: {}, Variation: {}'.format(systematic, variations))
        if systematic != 'nominal':
            return {}, {}
        for variation in variations:
            variation_suffix = '' if variation is None else variation
            key = systematic if variation is None else "{}{}".format(systematic, variation_suffix)
            
            all_bkg[key] = {}
            all_bkg_per_process[key] = {}

            for lepton_selection in lepton_selections:
                print(lepton_selection)
                directory = "{}/{}".format(systematic, lepton_selection) if variation is None else "{}{}/{}".format(
                    systematic, variation_suffix, lepton_selection
                )
                if root_file.GetDirectory(directory):
                    root_file.cd(directory)
                    current_dir = ROOT.gDirectory
                else:
                    print(f"Warning: Directory {directory} does not exist.")
                    continue
                
                all_bkg[key][lepton_selection] = None
                all_bkg_per_process[key][lepton_selection] = {}

                # print(current_dir.GetListOfKeys())
                
                for histo_key in current_dir.GetListOfKeys():
                    histo = histo_key.ReadObj()
                    histo.SetDirectory(0)
                    if not histo or not isinstance(histo, ROOT.TH1):
                        print(f"Invalid histogram detected: {histo}")
                        continue
                    if 'jet_composition' in kwargs.keys() and kwargs['jet_composition']:
                        process_histogram_bkg_jet_composition(histo, histo.GetName(), key, event_selection, lepton_selection, all_bkg, all_bkg_per_process, **kwargs)
                    else:
                        process_histogram_bkg(histo, histo.GetName(), key, event_selection, lepton_selection, all_bkg, all_bkg_per_process, **kwargs)

                    del histo

    return all_bkg, all_bkg_per_process

def merging_histo_data(root_file, event_selection, lepton_selections, **kwargs):
    all_data = {}

    for lepton_selection in lepton_selections:
        # print(lepton_selection)
        directory = "nominal/{}".format(lepton_selection)
        root_file.cd(directory)
        current_dir = ROOT.gDirectory

        all_data[lepton_selection] = None

        print('\nProcessing data...')
        for histo_key in current_dir.GetListOfKeys():
            histo = histo_key.ReadObj()
            process_histogram_data(histo, histo.GetName(), event_selection, lepton_selection, all_data, **kwargs)

    return all_data

def dataframe_for_jet_composition(root_file, lepton_selections, variable='mass_leading', year='', **kwargs):
    import pandas as pd
    dataframe_out = {
        "Process": [],
        "JetComposition": [],
        "Integral": [],
        "LeptonSelection": [],
    }
    summed_histograms = {}

    background_mapping = {
        'ttZJets': 'ttV',
        'tW': 'ST',
        'tbarW': 'ST',
        'ST': 'ST',
        'tt_dilepton': 'tt',
        'tt_semilepton': 'tt',
        'dy_': 'DY',
        # 'wj': 'DY',
        # 'dy_m-50': 'DY-m-50',
        # 'dy_m-10_50': 'DY-m-10_50',
        'WZ': 'VV',
        'ZZ': 'VV',
        'WW': 'VV',
        'qcd': 'QCD',
        'ttWJets': 'ttV',
        'ttZJets': 'ttV',
        'ttl': 'ttV',
        'tttt': '4t',
        'ttH_HTobb': 'ttH',
        'ttH_HToNonbb': 'ttH',
        'ttHH': 'ttX',
        # 'ttWH': 'ttX',
        # 'ttWW': 'ttX',
        # 'ttWZ': 'ttX',
        # 'ttZH': 'ttX',
        # 'ttZZ': 'ttX',
        'tttJ': 'tttX',
        'tttW': 'tttX',
    }
    jet_composition_mapping = {
        'pure_qcd': 'pure_qcd',
        'non_covered': 'others',
        'others': 'others',
        'hadronic_t': 'hadronic_t',
        'q_from_w_not_from_top': 'q_from_w_not_from_top',
        'q_from_w': 'q_from_w_from_top',
        'hadronic_w': 'hadronic_w',
        'b_from_top': 'b_from_top',
        'b_not_from_top': 'b_not_from_top',
        'q_from_w_plus_b': 'q_from_w_plus_b',
        'hadronic_w_not_from_t': 'hadronic_w_not_from_t',
    }

    process_pattern = re.compile(r"^(.*?)_hotvr_")
    jet_composition_pattern = re.compile(r"hotvr_(\w+)_"+variable)

    def filter_histogram(hist_name, kwargs):
        if 'additional_flag' in kwargs:
            additional_flag, in_name = kwargs['additional_flag']
            if in_name == 'in':
                if isinstance(additional_flag, list):
                    return any(flag in hist_name for flag in additional_flag)
                if additional_flag == 'scaling':
                    return not ('up' in hist_name or 'down' in hist_name)
                if additional_flag == 'leading_tag':
                    return 'subleading_tag' not in hist_name
                if additional_flag == 'subleading_tag':
                    return 'leading_and' not in hist_name
                return additional_flag in hist_name
            elif in_name == 'not_in':
                if isinstance(additional_flag, list):
                    return all(flag not in hist_name for flag in additional_flag)
                return additional_flag not in hist_name
            else:
                print(f"Wrong string for processing histo: {additional_flag}, {in_name}")
                sys.exit()
        return True

    print('\nDataframe for jet composition (bkgs)...')
    for lepton_selection in lepton_selections:
        print(lepton_selection)
        directory = f"nominal/{lepton_selection}"

        if root_file.GetDirectory(directory):
            root_file.cd(directory)
            current_dir = ROOT.gDirectory
        else:
            print(f"Warning: Directory {directory} does not exist.")
            continue

        for histo_key in current_dir.GetListOfKeys():
            obj = histo_key.ReadObj()
            if isinstance(obj, ROOT.TH1):
                hist_name = obj.GetName()

                if not filter_histogram(hist_name, kwargs=kwargs):
                    continue

                if 'dy_m-10_50_hotvr' in hist_name and '_lo' not in hist_name: #NLO DY at low mass
                        continue
                if year == '2022EE':
                    if 'dy_m-50' in hist_name:
                        continue
                if '202' in year:
                    if 'tW_dilepton'in hist_name or 'tbarW_dilepton' in hist_name:
                        continue

                process_match = process_pattern.match(hist_name)
                process = process_match.group(1) if process_match else None

                if process:
                    for bg_key, bg_name in background_mapping.items():
                        if bg_key in process:
                            process = bg_name
                            break
                if not process:
                    # print(f"Parsing process not done: {hist_name}")
                    continue 

                jet_composition_match = jet_composition_pattern.search(hist_name)
                jet_composition = jet_composition_match.group(1) if jet_composition_match else None

                if jet_composition and jet_composition in jet_composition_mapping:
                    jet_composition = jet_composition_mapping[jet_composition]

                if not jet_composition:
                    # print(f"Parsing jet composition not done: {hist_name}")
                    continue

                if 'tttW' in hist_name:
                    process = 'tttX'
                if 'ttWJ' in hist_name:
                    process = 'ttV'
                key = (process, jet_composition, lepton_selection)
                # if 'hadronic_t' in jet_composition:
                #     print(key, hist_name)
                if obj.Integral() < 0.0:
                    continue
                if process == 'ttH':
                    print(key, hist_name, obj.Integral())

                if key not in summed_histograms:
                    summed_histograms[key] = obj.Clone()
                else:
                    summed_histograms[key].Add(obj)

    for (process, jet_composition, lepton_selection), hist in summed_histograms.items():
        integral = hist.Integral()

        dataframe_out["Process"].append(process)
        dataframe_out["JetComposition"].append(jet_composition)
        dataframe_out["LeptonSelection"].append(lepton_selection)
        dataframe_out["Integral"].append(integral)

    return pd.DataFrame(dataframe_out)

def rebinning(h_target, h_variable_bins):
    for i in range(1, h_target.GetNbinsX() + 1):
        bin_content = h_target.GetBinContent(i)
        bin_error = h_target.GetBinError(i)
        bin_center = h_target.GetBinCenter(i)

        new_bin = h_variable_bins.FindBin(bin_center)
        
        h_variable_bins.SetBinContent(new_bin, h_variable_bins.GetBinContent(new_bin) + bin_content)

        current_error = h_variable_bins.GetBinError(new_bin)
        h_variable_bins.SetBinError(new_bin, ROOT.TMath.Sqrt(current_error**2 + bin_error**2))

    return h_variable_bins

def normalization(h_up, h_down, h_nom, type=''):
    # Use the max difference between up/down variation as norm. uncertainty
    tot_nom = h_nom.Integral()
    tot_up = h_up.Integral()
    tot_down = h_down.Integral()

    delta_up = (tot_up - tot_nom) / tot_nom
    delta_down = (tot_down - tot_nom) / tot_nom
    max_variation = max(abs(delta_up), abs(delta_down))
    # max_variation = max(abs(tot_nom-tot_up), abs(tot_nom-tot_down))

    print(f"!!! Normalization procedure of the systematic (FSR)!!!")
    print(f'Tot.Integral: {tot_nom} (nom); {tot_up} (up); {tot_down} (down); max. variation {max_variation}')

    # ratio_up = h_up.Clone("normalization_up")
    # ratio_down = h_down.Clone("normalization_down")
    # ratio_up.Reset()  # Keep the bin edges
    # ratio_down.Reset()

    for i in range(h_nom.GetNbinsX()):
        bin_err = h_nom.GetBinError(i+1)
        bin_val = h_nom.GetBinContent(i+1)

        if bin_err > 0 and bin_val > 0:
            # ratio_up.SetBinContent(i+1, 1+max_variation)
            # ratio_down.SetBinContent(i+1, 1-max_variation)
            h_up.SetBinContent(i + 1, bin_val * (1 + max_variation))
            h_down.SetBinContent(i + 1, bin_val * (1 - max_variation))
        else:
            continue

    return h_up, h_down
    # return ratio_up, ratio_down

def computing_tot_systematics(histos, lepton_selection='', systematics=[], scaling_factor=None, ):
    hist_nom = histos['nominal'][lepton_selection]
    n_bins = hist_nom.GetNbinsX()
    # print([histos[syst][lepton_selection].Integral() for syst in systematics if lepton_selection in histos[syst]])

    x, ex, y = array('d'), array('d'), array('d')
    ey_up, ey_down = array('d'), array('d')

    for ibin in range(1, n_bins + 1):  # ROOT bins start at 1
        sum_squares_up = 0.0
        sum_squares_down = 0.0

        nominal = hist_nom.GetBinContent(ibin)
        for systematic in systematics:
            if systematic == 'nominal':
                continue
            if not histos[systematic]:
                continue
            if lepton_selection not in histos[systematic]:
                continue

            val = histos[systematic][lepton_selection].GetBinContent(ibin)
            diff = val - nominal
            if 'Up' in systematic:
                sum_squares_up += diff ** 2
            elif 'Down' in systematic:
                sum_squares_down += diff ** 2

        total_up = math.sqrt(sum_squares_up)# + mc_stat2)
        total_down = math.sqrt(sum_squares_down)# + mc_stat2)

        x.append(hist_nom.GetBinCenter(ibin))
        ex.append(hist_nom.GetBinWidth(ibin) / 2)
        y.append(nominal)
        ey_up.append(total_up)
        ey_down.append(total_down)
        if scaling_factor:
            y = array('d', [val * scaling_factor for val in y])
            ey_up = array('d', [val * scaling_factor for val in ey_up])
            ey_down = array('d', [val * scaling_factor for val in ey_down])

    return x, y, ex, ey_down, ey_up 

def envelope_qcd_scale(histos, lepton_selections=[], flavors=[]):
    for region in ['transfer', 'prediction']:
        for process in PROCESSES:        
            for lepton_selection in lepton_selections:
                histos_qcd = []
                for qcd_sys in ['MEenv', 'MEfac', "MEren"]:
                    for variation in ['Up', 'Down']:
                        key_name = f''
                        
                        if histos[f'{region}_region_{process}_{qcd_sys}{variation}'] and lepton_selection in histos[f'{region}_region_{process}_{qcd_sys}{variation}'].keys():
                            print(histos[f'{region}_region_{process}_{qcd_sys}{variation}'], f'{region}_region_{process}_{qcd_sys}{variation}')
                            if histos[f'{region}_region_{process}_{qcd_sys}{variation}'][lepton_selection]:
                                histos_qcd.append(
                                    histos[f'{region}_region_{process}_{qcd_sys}{variation}'][lepton_selection]
                                )
                
                if len(histos_qcd) != 6: # no envelope method if env, fac and ren are not included
                    continue 

                if not histos[f'{region}_region_{process}_nominal']:
                    continue
                hist_nom = histos[f'{region}_region_{process}_nominal'][lepton_selection]
                
                hist_up = hist_nom.Clone(f'{hist_nom.GetName().replace("nominal", "QCDScaleUp")}')
                hist_up.Reset()
                hist_down = hist_nom.Clone(f'{hist_nom.GetName().replace("nominal", "QCDScaleDown")}')
                hist_down.Reset()
                for ibin in range(1, hist_nom.GetNbinsX()+1):
                    nominal = hist_nom.GetBinContent(ibin)
                    if nominal == 0:
                    # Handle zero-nominal case safely
                        bin_values = [h.GetBinContent(ibin) for h in histos_qcd]
                        hist_up.SetBinContent(ibin, max(bin_values))
                        hist_down.SetBinContent(ibin, min(bin_values))
                    else:
                        # print(f'non-zero nominal {nominal} {[h.GetBinContent(ibin) - nominal for h in histos_qcd]}')
                        deviations = [h.GetBinContent(ibin) - nominal for h in histos_qcd]

                        up_deviation = max(deviations)
                        down_deviation = min(deviations)

                        hist_up.SetBinContent(ibin, nominal + up_deviation)
                        hist_down.SetBinContent(ibin, nominal + down_deviation)

                # print(hist_nom.GetName(), hist_nom.Integral())
                # print(hist_up.GetName(), hist_up.Integral())
                # print(hist_down.GetName(), hist_down.Integral())
                key_up = f"{region}_region_{process}_QCDScaleUp"
                key_down = f"{region}_region_{process}_QCDScaleDown"

                histos.setdefault(key_up, {})
                histos.setdefault(key_down, {})
                if lepton_selection not in histos[key_up]:
                    histos[key_up][lepton_selection] = hist_up
                if lepton_selection not in histos[key_down]:
                    histos[key_down][lepton_selection] = hist_down
                
                for flavor in flavors:
                    histos_qcd = []
                    for qcd_sys in ['MEenv', 'MEfac', "MEren"]:
                        for variation in ['Up', 'Down']:
                            key_name = f''
                            
                            if histos[f'{region}_region_{process}_{flavor}_{qcd_sys}{variation}'] and lepton_selection in histos[f'{region}_region_{process}_{flavor}_{qcd_sys}{variation}'].keys():
                                if histos[f'{region}_region_{process}_{flavor}_{qcd_sys}{variation}'][lepton_selection]:
                                    histos_qcd.append(
                                        histos[f'{region}_region_{process}_{flavor}_{qcd_sys}{variation}'][lepton_selection]
                                    )

                    if len(histos_qcd) != 6: # no envelope method if env, fac and ren are not included
                        continue 
                    if not histos[f'{region}_region_{process}_{flavor}_nominal']:
                        continue
                    hist_nom = histos[f'{region}_region_{process}_{flavor}_nominal'][lepton_selection]
                    
                    hist_up = hist_nom.Clone(f'{hist_nom.GetName().replace("nominal", "QCDScaleUp")}')
                    hist_up.Reset()
                    hist_down = hist_nom.Clone(f'{hist_nom.GetName().replace("nominal", "QCDScaleDown")}')
                    hist_down.Reset()
                    for ibin in range(1, hist_nom.GetNbinsX()+1):
                        nominal = hist_nom.GetBinContent(ibin)
                        if nominal == 0:
                        # Handle zero-nominal case safely
                            bin_values = [h.GetBinContent(ibin) for h in histos_qcd]
                            if len(bin_values) == 0:
                                continue
                            hist_up.SetBinContent(ibin, max(bin_values))
                            hist_down.SetBinContent(ibin, min(bin_values))
                        else:
                            # print(f'non-zero nominal {nominal} {[h.GetBinContent(ibin) - nominal for h in histos_qcd]}')
                            deviations = [h.GetBinContent(ibin) - nominal for h in histos_qcd]
                            if not deviations:
                                continue

                            up_deviation = max(deviations)
                            down_deviation = min(deviations)

                            hist_up.SetBinContent(ibin, nominal + up_deviation)
                            hist_down.SetBinContent(ibin, nominal + down_deviation)

                    # print(hist_nom.GetName(), hist_nom.Integral())
                    # print(hist_up.GetName(), hist_up.Integral())
                    # print(hist_down.GetName(), hist_down.Integral())
                    key_up = f"{region}_region_{process}_{flavor}_QCDScaleUp"
                    key_down = f"{region}_region_{process}_{flavor}_QCDScaleDown"

                    histos.setdefault(key_up, {})
                    histos.setdefault(key_down, {})
                    if lepton_selection not in histos[key_up]:
                        histos[key_up][lepton_selection] = hist_up
                    if lepton_selection not in histos[key_down]:
                        histos[key_down][lepton_selection] = hist_down
    
    return histos
