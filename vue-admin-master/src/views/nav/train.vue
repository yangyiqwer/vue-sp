<template>
	<section>
		<!--工具条-->
		<el-row>
		<el-col :span="24" class="toolbar" style="padding-bottom: 0px;">
			<el-form :inline="true">
				<el-form-item>
				模型：
				</el-form-item>
				<el-form-item>
					<el-select v-model="model_value" placeholder="选择训练模型">
    					<el-option
     		 				v-for="item in model_options"
      						:key="item.model_value"
      						:label="item.model_label"
      						:value="item.model_value">
   		 				</el-option>
 					</el-select>
				</el-form-item>
				<el-form-item>
				训练集：
				</el-form-item>
				<el-form-item>
					<el-select v-model="data_value" placeholder="选择训练数据集">
    					<el-option
     		 				v-for="item in data_options"
        					:key="item.data_value"
        					:label="item.data_label"
        					:value="item.data_value">
         				</el-option>
        			</el-select>
        		</el-form-item>
        		<el-form-item>
        		模型参数保存名：
				</el-form-item>
				<el-form-item>
					<el-input v-model="para_save_name" placeholder="参数保存命名"></el-input>
				</el-form-item>
				<el-form-item>
					<el-button style="margin-left: 10px;"
					type="primary" :loading="istraining" @click="trainModel">训练</el-button>
				</el-form-item>
                <el-form-item>
                    <el-button style="margin-left: 10px;"
                    type="success" @click="getTrainlist">刷新</el-button>
                </el-form-item>
			</el-form>
		</el-col>
        <el-col :span="24">
    		<div id="trainingtabletitle" style="width:100%; text-align:center;">
                    <span style=" font-size:16px; line-height:3;">正在训练列表</span>
            </div>
        </el-col>
		<el-table :data="tarininglist" highlight-current-row v-loading="traininglistloading" style="width: 100%;">
            <el-table-column type="index" width="60">
            </el-table-column>
            <el-table-column prop="model" label="模型" width="150" sortable>
            </el-table-column>
            <el-table-column prop="dataset" label="数据集" width="200" sortable>
            </el-table-column>
            <el-table-column prop="modeldata_name" label="参数保存名" min-width="200" sortable>
            </el-table-column>
        </el-table>
        <el-col :span="24">
            <div id="finishtabletitle" style="width:100%; text-align:center;">
                    <span style=" font-size:16px; line-height:3;">训练完成列表</span>
            </div>
        </el-col>
        <el-table :data="finishlist" highlight-current-row v-loading="finishlistloading" style="width: 100%;">
            <el-table-column type="index" width="60">
            </el-table-column>
            <el-table-column prop="model" label="模型" width="150" sortable>
            </el-table-column>
            <el-table-column prop="dataset" label="数据集" width="200" sortable>
            </el-table-column>
            <el-table-column prop="modeldata_name" label="参数保存名" min-width="200" sortable>
            </el-table-column>
        </el-table>
        </el-row>
	</section>
</template>

<script>
	import echarts from 'echarts'
    import { serverbase } from '../../api/api';

  export default {
    data() {
      return {
        model_options: [],
        model_value: '',
        data_options: [],
        data_value: '',
        para_save_name: '',
        istraining: false,
        traininglistloading: false,
        tarininglist: [],
        finishlist: [],
        finishlistloading: false,
      }
    },
    methods: {
        getTrainlist(){
            this.traininglistloading = true;
            this.finishlistloading = true;
            fetch(serverbase() + 'train/' + '?name=' + encodeURIComponent('trainlist'))
            .then(response => response.json())
            .then(data => {
                this.tarininglist = data[0].traininglist;
                this.finishlist = data[0].finishlist;
                this.traininglistloading = false;
                this.finishlistloading = false;
            });
        },

    	trainModel(){
    		this.istraining = true;
            fetch(serverbase() + 'train/' + '?name=' + encodeURIComponent('train') + '&model=' + encodeURIComponent(this.model_value) + '&data=' + encodeURIComponent(this.data_value) + '&save=' + encodeURIComponent(this.para_save_name))
            .then(response => response.json())
            .then(data => {
                this.getTrainlist();
                this.istraining = false;
            });
    	},

    	getModelList(){
    		fetch(serverbase() + 'train/' + '?name=' + encodeURIComponent('modelset'))
    		.then(response => response.json())
  			.then(data => {
  				this.model_options = data;
    		});
    	},

    	getTraindatasetList(){
    		fetch(serverbase() + 'train/' + '?name=' + encodeURIComponent('dataset'))
    		.then(response => response.json())
  			.then(data => {
  				this.data_options = data;
    		});
    	},
    },

    mounted: function () {
    	this.getModelList();
    	this.getTraindatasetList();
        this.getTrainlist();
    },
    updated: function () {
    }
  }
</script>

<style scoped>
	.chart-container {
        width: 100%;
        float: left;
    }
</style>