using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestSoftsign
    {
        [TestMethod]
        public void Test_Softsign_Execute()
        {
            softsign = new SoftsignLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            softsign.SetInput(data);

            softsign.Execute();

            Data2D output = softsign.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], SoftsignFunc(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], SoftsignFunc(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], SoftsignFunc(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], SoftsignFunc(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], SoftsignFunc(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], SoftsignFunc(-3.0), 0.00000001);
        }

        private double SoftsignFunc(double x)
        {
            return x / (1 + Math.Abs(x));
        }

        [TestMethod]
        public void Test_SoftSign_KerasModel()
        {
            string path = Resources.TestsFolder + "test_softsign_model.json";
            var reader = new ReaderKerasModel(path);

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 8, 1, 1);

            inp[0, 0, 0, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 2, 0, 0] = -1;
            inp[0, 3, 0, 0] = 0;

            inp[0, 4, 0, 0] = 3;
            inp[0, 5, 0, 0] = 1;
            inp[0, 6, 0, 0] = 1;
            inp[0, 7, 0, 0] = 2;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 4);
            Assert.AreEqual(ou.GetDimension().w, 1);

            Assert.AreEqual(ou[0, 0, 0, 0], -0.5, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], 0.8461538553237915, 0.00001);
            Assert.AreEqual(ou[0, 0, 2, 0], 0.9487179517745972, 0.00001);
            Assert.AreEqual(ou[0, 0, 3, 0], 0.8999999761581421, 0.00001);
        }

        private SoftsignLayer softsign;
    }
}
