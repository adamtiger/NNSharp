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
    public class TestReshape2D
    {
        [TestMethod]
        public void Test_Reshape2D_Execute()
        {
            Data2D data = new Data2D(3, 3, 2, 1);

            for (int h = 0; h < 3; ++h)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 2; ++c)
                    {
                        data[h, w, c, 0] = h * 6 + w * 2 + c + 1;
                    }
                }
            }

            Reshape2DLayer res = new Reshape2DLayer(3, 2, 3, 1);
            res.SetInput(data);
            res.Execute();
            Data2D output = res.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 3);
            Assert.AreEqual(dim.h, 3);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 1.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], 2.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 2, 0], 3.0, 0.0000001);

            Assert.AreEqual(output[0, 1, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 2, 0], 6.0, 0.0000001);

            Assert.AreEqual(output[1, 0, 0, 0], 7.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 1, 0], 8.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 2, 0], 9.0, 0.0000001);

            Assert.AreEqual(output[1, 1, 0, 0], 10.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 1, 0], 11.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 2, 0], 12.0, 0.0000001);

            Assert.AreEqual(output[2, 0, 0, 0], 13.0, 0.0000001);
            Assert.AreEqual(output[2, 0, 1, 0], 14.0, 0.0000001);
            Assert.AreEqual(output[2, 0, 2, 0], 15.0, 0.0000001);

            Assert.AreEqual(output[2, 1, 0, 0], 16.0, 0.0000001);
            Assert.AreEqual(output[2, 1, 1, 0], 17.0, 0.0000001);
            Assert.AreEqual(output[2, 1, 2, 0], 18.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Reshape2D_Null_Input()
        {
            Data2D data = null;
            Reshape2DLayer res = new Reshape2DLayer(1, 2, 1, 4);
            res.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Reshape2D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Reshape2DLayer res = new Reshape2DLayer(1, 8, 2, 1);
            res.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Reshape2D_WrongSIzes()
        {
            Data2D data = new Data2D(2, 3, 5, 2);
            Reshape2DLayer res = new Reshape2DLayer(1, 2, 1, 4);
            res.SetInput(data);
        }

        [TestMethod]
        public void Test_Reshape2D_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_reshape_model.json";
            string pathInput = Resources.TestsFolder + "test_reshape_input.json";
            string pathOutput = Resources.TestsFolder + "test_reshape_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}
