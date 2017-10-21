using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestCropping1D
    {
        [TestMethod]
        public void Test_Cropping1D_Execute()
        {
            Data2D data = new Data2D(1, 6, 2, 1);

            int cntr = 0;

            for (int w = 0; w < 6; ++w)
            {
                for (int c = 0; c < 2; ++c)
                {
                    cntr += 1;
                    data[0, w, c, 0] = cntr;
                }
            }


            Cropping1DLayer crop = new Cropping1DLayer(1, 2);
            crop.SetInput(data);
            crop.Execute();
            Data2D output = crop.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 3);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], data[0, 1, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], data[0, 1, 1, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], data[0, 2, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], data[0, 2, 1, 0], 0.0000001);
            Assert.AreEqual(output[0, 2, 0, 0], data[0, 3, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 2, 1, 0], data[0, 3, 1, 0], 0.0000001);

        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping1D_Null_Input()
        {
            Data2D data = null;
            Cropping1DLayer crop = new Cropping1DLayer(1, 2);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping1D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Cropping1DLayer crop = new Cropping1DLayer(2, 2);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping1D_Negative_Trim()
        {
            Data2D data = new Data2D(1, 4, 3, 5);
            Cropping1DLayer crop = new Cropping1DLayer(4, -5);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping1D_Not1DInput_Trim()
        {
            Data2D data = new Data2D(2, 4, 3, 5);
            Cropping1DLayer crop = new Cropping1DLayer(1, 1);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping1D_TooMuchCropping_Trim()
        {
            Data2D data = new Data2D(1, 4, 3, 5);
            Cropping1DLayer crop = new Cropping1DLayer(2, 3);
            crop.SetInput(data);
        }

        [TestMethod]
        public void Test_Cropping1D_KerasModel()
        {
            string path = Resources.TestsFolder + "test_crop_1D_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 5, 2, 1);

            for (int l = 0; l < 5; ++l)
            {
                inp[0, l, 0, 0] = l + 1;
                inp[0, l, 1, 0] = -(l + 1);
            }

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 2);

            Assert.AreEqual(ou[0, 0, 0, 0], 2.0, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], -2.0, 0.00001);
            Assert.AreEqual(ou[0, 1, 0, 0], 3.0, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], -3.0, 0.00001);
        }
    }
}
