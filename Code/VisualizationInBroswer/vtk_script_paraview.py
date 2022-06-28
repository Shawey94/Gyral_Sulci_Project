'''
Script to Save Paraview Plots in a Batch
Paraview Version 5.8.1
Author: Lin Zhao
'''

# Import the Simple Module from the Paraview and Disable Automatic Camera Reset on 'Show'
from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()


class Paraview_Batch_Plot():
    def __init__(self, subj_list, pdata_name, vtk_path, png_path):
    
        self.subj_list = subj_list
        self.pdata_name = pdata_name
        self.vtk_path = vtk_path
        self.png_path = png_path

        # Create a new 'Render View' for Sagittal View
        self.renderView1 = FindViewOrCreate('self.renderView1', viewtype='RenderView')
        # Modify the ViewSize if Needed
        self.renderView1.ViewSize = [430, 400]

        # Create a new 'Render View' for Axial View
        self.renderView2 = FindViewOrCreate('self.renderView2', viewtype='RenderView')
        # Modify the ViewSize if Needed
        self.renderView2.ViewSize = [300, 400]

        # Create a new 'Render View' for Coronal View
        self.renderView3 = FindViewOrCreate('self.renderView3', viewtype='RenderView')
        # Modify the ViewSize if Needed
        self.renderView3.ViewSize = [350, 400]

    def main(self):
        for sub in self.subj_list:
            # Create a new 'Legacy VTK Reader'
            fname = [self.vtk_path+str(sub)+'.vtk']
            print(fname)
            brain_vtk = LegacyVTKReader(FileNames=fname)
            
            # Set active Source
            SetActiveSource(brain_vtk)

            for pdata in self.pdata_name:
            
                Custom_Pdata = pdata
                print(Custom_Pdata)

                # Get Color Transfer Function/Color Map for 'Custom_Array'
                Tmp_LUT = GetColorTransferFunction(Custom_Pdata)
                # Get Opacity Transfer Function/Opacity Map
                #Tmp_PWF = GetOpacityTransferFunction(Custom_Pdata)
                
                
                # Apply the Preset. 
                Tmp_LUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.3362075090408325, 0.865003, 0.865003, 0.865003, 0.672415018081665, 0.705882, 0.0156863, 0.14902]
                Tmp_LUT.ScalarRangeInitialized = 1.0

                # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
                Tmp_LUT.ApplyPreset('Cool to Warm', True)
                #Tmp_LUT.ApplyPreset('jet', True)   
                
                # Show Data in Sagittal View
                sagittal_vtkDisplay = Show(brain_vtk, self.renderView1, 'GeometryRepresentation')
                
                # Traced defaults for the display properties.
                sagittal_vtkDisplay.Representation = 'Surface'
                sagittal_vtkDisplay.ColorArrayName = ['POINTS', Custom_Pdata]
                sagittal_vtkDisplay.LookupTable = Tmp_LUT
                sagittal_vtkDisplay.OSPRayScaleArray = Custom_Pdata
                sagittal_vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
                sagittal_vtkDisplay.SelectOrientationVectors = Custom_Pdata
                sagittal_vtkDisplay.ScaleFactor = 17.09222869873047
                sagittal_vtkDisplay.SelectScaleArray = Custom_Pdata
                sagittal_vtkDisplay.GlyphType = 'Arrow'
                sagittal_vtkDisplay.GlyphTableIndexArray = Custom_Pdata
                sagittal_vtkDisplay.GaussianRadius = 0.8546114349365235
                sagittal_vtkDisplay.SetScaleArray = ['POINTS', Custom_Pdata]
                sagittal_vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
                sagittal_vtkDisplay.OpacityArray = ['POINTS', Custom_Pdata]
                sagittal_vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
                sagittal_vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
                sagittal_vtkDisplay.PolarAxes = 'PolarAxesRepresentation'

                # Init the 'PiecewiseFunction' Selected for 'ScaleTransferFunction'
                sagittal_vtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Init the 'PiecewiseFunction' Selected for 'OpacityTransferFunction'
                sagittal_vtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Hide Color Bar/Color Legend
                sagittal_vtkDisplay.SetScalarBarVisibility(self.renderView1, False)



                # Show Data in Axial View
                axial_vtkDisplay = Show(brain_vtk, self.renderView2, 'GeometryRepresentation')
                
                # Traced defaults for the display properties.
                axial_vtkDisplay.Representation = 'Surface'
                axial_vtkDisplay.ColorArrayName = ['POINTS', Custom_Pdata]
                axial_vtkDisplay.LookupTable = Tmp_LUT
                axial_vtkDisplay.OSPRayScaleArray = Custom_Pdata
                axial_vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
                axial_vtkDisplay.SelectOrientationVectors = Custom_Pdata
                axial_vtkDisplay.ScaleFactor = 17.09222869873047
                axial_vtkDisplay.SelectScaleArray = Custom_Pdata
                axial_vtkDisplay.GlyphType = 'Arrow'
                axial_vtkDisplay.GlyphTableIndexArray = Custom_Pdata
                axial_vtkDisplay.GaussianRadius = 0.8546114349365235
                axial_vtkDisplay.SetScaleArray = ['POINTS', Custom_Pdata]
                axial_vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
                axial_vtkDisplay.OpacityArray = ['POINTS', Custom_Pdata]
                axial_vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
                axial_vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
                axial_vtkDisplay.PolarAxes = 'PolarAxesRepresentation'

                # Init the 'PiecewiseFunction' Selected for 'ScaleTransferFunction'
                axial_vtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Init the 'PiecewiseFunction' Selected for 'OpacityTransferFunction'
                axial_vtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Hide Color Bar/Color Legend
                axial_vtkDisplay.SetScalarBarVisibility(self.renderView2, False)
                
                
                # Show Data in Coronal View
                coronal_vtkDisplay = Show(brain_vtk, self.renderView3, 'GeometryRepresentation')
                
                # Traced defaults for the display properties.
                coronal_vtkDisplay.Representation = 'Surface'
                coronal_vtkDisplay.ColorArrayName = ['POINTS', Custom_Pdata]
                coronal_vtkDisplay.LookupTable = Tmp_LUT
                coronal_vtkDisplay.OSPRayScaleArray = Custom_Pdata
                coronal_vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
                coronal_vtkDisplay.SelectOrientationVectors = Custom_Pdata
                coronal_vtkDisplay.ScaleFactor = 17.09222869873047
                coronal_vtkDisplay.SelectScaleArray = Custom_Pdata
                coronal_vtkDisplay.GlyphType = 'Arrow'
                coronal_vtkDisplay.GlyphTableIndexArray = Custom_Pdata
                coronal_vtkDisplay.GaussianRadius = 0.8546114349365235
                coronal_vtkDisplay.SetScaleArray = ['POINTS', Custom_Pdata]
                coronal_vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
                coronal_vtkDisplay.OpacityArray = ['POINTS', Custom_Pdata]
                coronal_vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
                coronal_vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
                coronal_vtkDisplay.PolarAxes = 'PolarAxesRepresentation'

                # Init the 'PiecewiseFunction' Selected for 'ScaleTransferFunction'
                coronal_vtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Init the 'PiecewiseFunction' Selected for 'OpacityTransferFunction'
                coronal_vtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.19266299903392792, 1.0, 0.5, 0.0]
                # Hide Color Bar/Color Legend
                coronal_vtkDisplay.SetScalarBarVisibility(self.renderView3, False)


                # Hide Orientation Axes and Center Axes
                self.renderView1.OrientationAxesVisibility = 0
                self.renderView1.CenterAxesVisibility = 0
                # Current Camera Placement for self.renderView1
                self.renderView1.CameraPosition = [-328.6176531321803, -17.45416259765625, 15.97125244140625]
                self.renderView1.CameraFocalPoint = [0.6561470031738281, -17.45416259765625, 15.97125244140625]
                self.renderView1.CameraViewUp = [0.0, 0.0, 1.0]
                self.renderView1.CameraParallelScale = 124.77401412649962

                # Hide Orientation Axes and Center Axes
                self.renderView2.OrientationAxesVisibility = 0
                self.renderView2.CenterAxesVisibility = 0
                # Current Camera Placement for self.renderView2
                self.renderView2.CameraPosition = [0.6561470031738281, -17.45416259765625, 367.49659366891154]
                self.renderView2.CameraFocalPoint = [0.6561470031738281, -17.45416259765625, 15.97125244140625]
                self.renderView2.CameraParallelScale = 133.79088624110994

                # Hide Orientation Axes and Center Axes
                self.renderView3.OrientationAxesVisibility = 0
                self.renderView3.CenterAxesVisibility = 0
                # Current Camera Placement for self.renderView3
                self.renderView3.CameraPosition = [0.6561470031738281, -346.7279627330103, 15.97125244140625]
                self.renderView3.CameraFocalPoint = [0.6561470031738281, -17.45416259765625, 15.97125244140625]
                self.renderView3.CameraViewUp = [0.0, 0.0, 1.0]
                self.renderView3.CameraParallelScale = 124.77401412649962


                # Save Screenshot
                SaveScreenshot(self.png_path+'/'+str(sub)+'_'+str(pdata)+'_01.png', 
                    self.renderView1,
                    OverrideColorPalette='PrintBackground', 
                    CompressionLevel='0')
                    
                SaveScreenshot(self.png_path+'/'+str(sub)+'_'+str(pdata)+'_02.png', 
                    self.renderView2,
                    OverrideColorPalette='PrintBackground', 
                    CompressionLevel='0')
                    
                SaveScreenshot(self.png_path+'/'+str(sub)+'_'+str(pdata)+'_03.png', 
                    self.renderView3,
                    OverrideColorPalette='PrintBackground', 
                    CompressionLevel='0')
                    
            # Destroy brain_vtk
            Delete(brain_vtk)
            del brain_vtk


subj_list = ['148941_gyri_nozscore','148941_sulci_nozscore']
pdata = ['label'+"%d" % j for j in range(0,100)]

vtk_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/generated_vtks/'
png_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/PngsFromParaview'
new_plot = Paraview_Batch_Plot(subj_list,pdata,vtk_path,png_path)
new_plot.main()

'''
subj_list = range(1,10)
pdata = ['consistent_3hinges_region']
new_plot = Paraview_Batch_Plot(subj_list,pdata,'D:\\K_100_L_0.11\\3hinge\\','D:/K_100_L_0.11/3hinge/plots/')
new_plot.main()
'''
