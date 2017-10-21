using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialLayers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestELu
    {

        [TestMethod]
        public void Test_ELu_Execute()
        {
            double alpha = 1.0;
            elu = new ELuLayer(alpha);

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            elu.SetInput(data);

            elu.Execute();

            Data2D output = elu.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], 4.0, 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], 2.0, 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], alpha*(Math.Exp(-2) - 1), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], 3.0, 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], alpha * (Math.Exp(-1) - 1), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], alpha * (Math.Exp(-3) - 1), 0.00000001);
        }

        [TestMethod]
        public void Test_ELu_KerasModel()
        {
            string path = Resources.TestsFolder + "test_elu_model.json";
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

            Assert.AreEqual(ou[0, 0, 0, 0], -0.6321205496788025, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], 5.5, 0.00001);
            Assert.AreEqual(ou[0, 0, 2, 0], 18.5, 0.00001);
            Assert.AreEqual(ou[0, 0, 3, 0], 9.0, 0.00001);
        }

        private ELuLayer elu;
    }
}
