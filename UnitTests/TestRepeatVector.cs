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
    public class TestRepeatVector
    {
        [TestMethod]
        public void Test_RepeatVector_Execute()
        {
            Data2D data = new Data2D(1, 1, 4, 2);

            for (int i = 0; i < 4; ++i)
            {
                data[0, 0, i, 0] = 2 * i + 1;
                data[0, 0, i, 1] = -(2 * i + 1);
            }

            RepeatVectorLayer res = new RepeatVectorLayer(3);
            res.SetInput(data);
            res.Execute();
            Data2D output = res.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 2);
            Assert.AreEqual(dim.c, 4);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 3);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 1.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 2, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 3, 0], 7.0, 0.0000001);

            Assert.AreEqual(output[0, 1, 0, 0], 1.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 2, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 3, 0], 7.0, 0.0000001);

            Assert.AreEqual(output[0, 2, 0, 0], 1.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 2, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 3, 0], 7.0, 0.0000001);

            Assert.AreEqual(output[0, 0, 0, 1], -1.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 1], -3.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 2, 1], -5.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 3, 1], -7.0, 0.0000001);

            Assert.AreEqual(output[0, 1, 0, 1], -1.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 1], -3.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 2, 1], -5.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 3, 1], -7.0, 0.0000001);

            Assert.AreEqual(output[0, 2, 0, 1], -1.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 1, 1], -3.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 2, 1], -5.0, 0.0000001);
            Assert.AreEqual(output[0, 2, 3, 1], -7.0, 0.0000001);


        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_RepeatVector_Null_Input()
        {
            Data2D data = null;
            RepeatVectorLayer rep = new RepeatVectorLayer(4);
            rep.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_RepeatVector_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            RepeatVectorLayer rep = new RepeatVectorLayer(8);
            rep.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_RepeatVector_WrongSizesHeight()
        {
            Data2D data = new Data2D(2, 1, 5, 2);
            RepeatVectorLayer rep = new RepeatVectorLayer(3);
            rep.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_RepeatVector_WrongSizesWidth()
        {
            Data2D data = new Data2D(1, 3, 5, 2);
            RepeatVectorLayer rep = new RepeatVectorLayer(3);
            rep.SetInput(data);
        }

        [TestMethod]
        public void Test_RepeatVector_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_repeatvector_model.json";
            string pathInput = Resources.TestsFolder + "test_repeatvector_input.json";
            string pathOutput = Resources.TestsFolder + "test_repeatvector_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}
