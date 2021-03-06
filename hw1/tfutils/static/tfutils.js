
// TfUtils namespace
var TfUtils = (function() {
    var dictTest = {};
    var dictTask = {};
    
    var jTestResults = $("<span/>");

    function getTestHolder() {
	return $("div#test_results");
    }
    
    function getTaskHolder() {
	return $("div#task_results");
    }

    function buildTest(dictTestProperties) {
	var sName = dictTestProperties.name
	var jHolder = getTestHolder();
	var jTest = $("<div class='test'/>");
	var jHeader = $("<div class='test_header'/>");
	var jNameHolder = $("<div class='test_nameholder'/>");
	var jExpandTb = $("<div class='test_expand collapsed'/>");
	var jExpandConsole = $("<div class='test_expand collapsed'/>");
	var jRunTestButton = $("<div class='test_button'/>");
	var jTestContent = $("<div class='test_content collapsed'/>");
	var jPre = $("<pre class='test_pre code'/>");
	jHeader.attr("title",dictTestProperties.description);
	jNameHolder.text(sName.split('.')[1]);
	jHeader.append(jNameHolder);
	jExpandTb.append($("<a href=''/>").text("Show Failure"));
	jHeader.append(jExpandTb);
	jExpandConsole.append($("<a href=''/>").text("Show Output"));
	jHeader.append(jExpandConsole);
	jHeader.append(jRunTestButton);
	jHeader.append($("<div class='clear'/>"));
	jTestContent.append(jPre);
	jTest.append(jHeader);
	jTest.append(jTestContent);
	jHolder.append(jTest);

	function prep() {
	    jRunTestButton.unbind('click');
	    jTest.addClass("running");
	    if (t.showing) {
		jTestContent.slideUp();
		t.showing = null;
	    }
	}

	function cb(event) {
	    prep();
	    runTest(sName);
	}
	jTest.click(cb);
	jRunTestButton.attr("title","Click to run this test.");

	function buildShowText(sKey) {
	    return function(event) {
		if (sKey == t.showing) {
		    jTestContent.slideUp();
		    t.showing = null;
		} else {
		    jPre.text(t[sKey]);
		    if (t.showing === null) {
			jTestContent.slideDown();
		    }
		    t.showing = sKey;
		}
		event.stopPropagation();
		event.preventDefault();
	    }
	}

	function setShowButton(sKey, jButton) {
	    return function(sMsg) {
		t[sKey] = sMsg;
		if (sMsg) {
		    jButton.show();
		} else {
		    jButton.hide();
		}	    
	    }
	}
	jExpandTb.find('a').click(buildShowText("traceback"));
	jExpandConsole.find('a').click(buildShowText("console"));

	var t = {
	    name: sName,
	    j: jTest,
	    jButton: jRunTestButton,
	    result: null,
	    prep: prep,
	    cb: cb,
	    traceback: "",
	    console: "",
	    showing: null,
	    setTraceback: setShowButton("traceback",jExpandTb),
	    setConsole: setShowButton("console", jExpandConsole)
	};

	dictTest[sName] = t;
	setTestResult(sName,dictTestProperties.result);
    }

    function buildTask(dictTaskProperties) {
	var sName = dictTaskProperties.name;
	var sId = dictTaskProperties.id
	var tk = {
	    name: sName,
	    id: sId
	};
	var jTask = $("<div class='task'/>");
	var jHeader = $("<div class='task_header'/>");
	var jDescription = $("<div class='task_description'/>");
	var jTitleHolder = $("<div class='task_title_holder'/>");
	var jTaskContent = $("<div class='task_content collapsed'/>");
	var jError = $("<pre class='task_error collapsed code'/>");
	var jConsole = $("<pre class='task_console collapsed code'/>");
	var jRunHolder = $("<div class='task_run_holder'/>");
	
	jTitleHolder.text(sName);
	
	jRunHolder.append($("<a href=''/>").text("Run"));
	jHeader.append(jTitleHolder);
	jHeader.append(jRunHolder);

	var jDisplay = null;
	if (dictTaskProperties.type == "graph") {
	    jDisplay = $("<canvas width='480' height='320' class='graph'/>");
	} else if (dictTaskProperties.type == "chart") {
	    jDisplay = $("<div id='" + sId + "_chart'/>");
	}
	jTaskContent.append(jDisplay);

	jTask.append(jHeader);
	jTask.append($("<div class='clear'/>"));
	if (dictTaskProperties.description) {
	    jDescription.text(dictTaskProperties.description);
	    jTask.append(jDescription);
	}
	jTask.append(jError);
	jTask.append(jConsole);
	jTask.append(jTaskContent);

	tk.display = jDisplay;
	tk.j = jTask;
	function handleErrors(fxn) {
	    return function(json) {
		if (json.console) {
		    jConsole.text(json.console);
		    jConsole.slideDown();
		} else {
		    jConsole.slideUp();
		}
		if (json.tb) {
		    jError.text(json.tb);
		    jError.slideDown();
		} else {
		    jError.slideUp();		
		}
		
		if (json.valid !== false) {
		    var ret = fxn(json.result);
		    if (ret) {
			jTaskContent.slideDown();
		    }
		}
	    }
	}
	tk.cb = (function() {
	    sType = dictTaskProperties.type;
	    if (sType == "graph") {
		return handleErrors(function(result) {
		    var iWidth = jTask.width();
		    var iHeight = Math.floor(iWidth*9.0/16.0);
		    jDisplay.attr("width",iWidth);
		    jDisplay.attr("height", iHeight);
		    var graph = buildSpringyGraph(result);
		    drawSpringyGraph(graph,jDisplay);
		    return graph;
		});
	    } else if (sType == "chart") {
		return handleErrors(function(dictChart) {
		    var iWidth = jTask.width();
		    var iHeight = Math.floor(iWidth*9.0/16.0);
		    dictChart.chart.renderTo = sId + "_chart";
		    dictChart.chart.width = iWidth;
		    dictChart.chart.height = iHeight;
		    var chart = new Highcharts.Chart(dictChart);
		    return chart;
		});
	    }
	})();
	tk.button = jRunHolder.find('a');
	tk.button.click(function(event) {
	    event.preventDefault();
	    jTaskContent.slideUp();
	    jTask.addClass("running");
	    jRunHolder.find('a').hide();
	    runTask(sId);
	});

	dictTask[sId] = tk;
	getTaskHolder().append(jTask);
    }

    function buildSpringyGraph(listEdges) {
	var setNode = {};
	var graph = new Graph();
	function addNode(sName) {
	    if (setNode[sName] === undefined) {
		var node = graph.newNode({label: sName});
		setNode[sName] = node;
		return node;
	    }
	    return setNode[sName]
	}
	$.each(listEdges, function(_,listE) {
	    var nodeSrc = addNode(listE[0]);
	    var nodeDest = addNode(listE[1]);
	    graph.newEdge(nodeSrc, nodeDest, {color: '#FF0000'});
	});
	return graph;
    }

    function drawSpringyGraph(graph,canvas) {
	$(canvas).springy(graph);
    }

    function runTask(sTask) {
	var tk = dictTask[sTask];
	function wrapper(json) {
	    tk.j.removeClass("running");
	    tk.button.show();
	    tk.cb(json);
	}
	$.post("/task/" + tk.id + "/", null, wrapper, "json");
    }

    function runTest(sTest) {
	$("div#run_all").hide();
	$.post("/test/run/", {"tests":sTest}, showTestResults, "json");
    }

    function showTestResults(json) {
	$.each(json, function(ix,d) {
	    var t = dictTest[d.name];
	    setTestResult(d.name, d.results.result,
			  d.results.failures.join("\n\n"),
			  d.results.console);
	});
	$("div#run_all").show();
    }

    function setTestResult(sTest,nResult,sTb,sConsole) {
	var t = dictTest[sTest];
	t.result = nResult;
	var jTest = t.j;
	var jButton = t.jButton;
	setTestButtonColor(jButton, nResult);
	t.setTraceback(sTb);
	t.setConsole(sConsole);
	updateTestResults();
	t.j.removeClass("running");
	jButton.click(t.cb);
    }
    
    function setTestButtonColor(jButton,nResult) {
	$.each(["test_success", "test_failure", "test_unknown"],
	       function(ix,sClass) {jButton.removeClass(sClass);});
	if (nResult === true) {
	    jButton.addClass("test_success");
	} else if (nResult === false) {
	    jButton.addClass("test_failure");
	} else {
	    jButton.addClass("test_unknown");
	}
    }

    function buildRunAllButton() {
	var jButton = $("<a href=''/>").text("Run All");
	var jHolder = $("<div id='run_all' />");
	jHolder.append(jButton);
	function cb(event) {
	    var listToRun = [];
	    $.each(dictTest, function(_,t) {
		listToRun.push(t.name);
		t.prep();
	    });
	    runTest(listToRun.join(','));
	    event.preventDefault();
	}
	jHolder.find('a').click(cb);
	$("div#run_all_holder").prepend(jHolder)
	    .append($("<div class='clear'/>"));

    }

    function loadInitialData(fxnCb) {
	var dictData = 	{};
	var cFound = 0;
	var cMaxRequests = 2;
	function loadMetadata(json) {
	    dictData.sTaskTitle = json.sTaskTitle;
	    dictData.sTaskSubtitle = json.sTaskSubtitle;
	    loadTasks(json.listTask);
	    cFound++;
	    if (cFound >= cMaxRequests) {
		return fxnCb(dictData);
	    }
	}
	function loadTests(json) {
	    dictData.listTest = json;
	    cFound++;
	    if (cFound >= cMaxRequests) {
		return fxnCb(dictData);
	    }
	}
	function loadTasks(json) {
	    json.sort(function(t1,t2) {
		return t1.priority - t2.priority;
	    });
	    $.each(json,function(_,dictTask) {
		buildTask(dictTask);
	    });
	}
	$.get("/metadata/",null,loadMetadata, "json");
	$.get("/test/load/",null,loadTests, "json");
    }

    function updateTestResults() {
	var c = 0;
	var cTotal = 0;
	$.each(dictTest, function(_,t) {
	    if (t.result) {
		c++;
	    }
	    cTotal++;
	});
	jTestResults.text(" (" + c + "/" + cTotal + ")");
    }

    function setTitle(sTitle, sSubtitle) {
	$(document).attr("title", sTitle + " - " + sSubtitle);
	var jSpanTitle = $("<span/>").text(sTitle);
	var jSpanSubtitle = $("<span/>").text(sSubtitle);
	$("div#right_title").append(jSpanTitle).append($("<br/>"))
						       .append(jSpanSubtitle);
    }

    function setInitialUi(dictData) {
	setTitle(dictData.sTaskTitle, dictData.sTaskSubtitle);
	$.map(dictData.listTest, buildTest);
    }

    function load() {
	$("div#tabs").tabs();
	loadInitialData(setInitialUi);
	$("span#tab_test_lbl").append(jTestResults);
	buildRunAllButton()
    }

    return {load: load};
})();

$(document).ready(TfUtils.load);