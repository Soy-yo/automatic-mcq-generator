const submit = () => {
    const submitButton = $("#submit");
    const originalText = submitButton.html();
    submitButton.prop("disabled", true);
    submitButton.html("Awaiting server response...");

    (async () => {
        $("#resultContainer").removeClass("d-none").empty();
        try {
            const text = $("#documents").val();
            const questions = parseInt($("#questionsPerDoc").val());
            const arguments = getArgs();
            // Override with other input if necessary
            arguments.questions_per_document = questions;
            const response = await fetch('/documents/', {
                method: 'POST',
                body: JSON.stringify({ text, arguments }),
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
            });
            const data = await response.json();
            showResults(data);
        } catch (e) {
            console.log(e);
        } finally {
            submitButton.html(originalText);
            submitButton.prop("disabled", false);
        }
    })();
};

const showResults = results => {
    const resultDivs = results.map(createMCQs);
    const titleRow = $("<div></div>").attr({class: "row align-baseline"});
    const titleCol = $("<div></div>")
        .attr({class: "col"})
        .appendTo(titleRow);
    const downloadBtnCol = $("<div></div>")
        .attr({id: "downloadBtnsCol", class: "col text-end"})
        .appendTo(titleRow);

    $("<h2></h2>").text("Results").appendTo(titleCol);
    $("<button></button>")
        .text("Generate file")
        .attr({type: "button", class: "btn btn-primary"})
        .click(downloadQuestions)
        .appendTo(downloadBtnCol);

    $("#resultContainer").append(titleRow, ...resultDivs);
};

const createMCQs = (result, i) => {
    const contextId = `context-${i}`;
    const wrapper = $("<div></div>");

    const contextRow = $("<div></div>")
        .attr({class: "row"});

    const contextGroup = $("<div></div>")
        .attr({class: "col form-group"});
    const contextLabel = $("<label></label>")
        .text("Context")
        .attr({for: contextId});
    const contextOutput = $("<div></div>")
        .text(result.document)
        .attr({
            id: contextId,
            class: "text-secondary font-italic bg-light " +
                "border border-secondary rounded p-3",
            name: "context"
        });

    contextGroup.append(contextLabel, contextOutput);
    contextRow.append(contextGroup);
    wrapper.append(contextRow);

    const mcqs = result.mcqs.map((mcq, j) => createOneMCQ(mcq, i, j));

    for (let i = 0; i < mcqs.length; i += 2) {
        const questionRow = $("<div></div>")
            .attr({class: "row"});
        [mcqs[i], mcqs[i + 1]].map(mcq =>
            $("<div></div>")
                .attr({class: "col"})
                .append(mcq)
                .appendTo(questionRow)
        );
        wrapper.append(questionRow);
    }

    return wrapper;
};

const createOneMCQ = (mcq, i, j) => {
    const form = $("<form></form>").attr({id: `mcq-${j}`});
    const questionId = `question-${i}-${j}`;

    const questionGroup = $("<div></div>")
        .attr({class: "form-group mb-3"});
    const questionLabel = $("<label></label>")
        .text(`Question ${j + 1}`)
        .attr({for: questionId, class: "font-weight-bold"});
    const questionInput = $("<input>")
        .val(mcq.question)
        .attr({
            id: questionId,
            type: "text",
            class: "form-control border-primary",
            name: `question`
        });

    const labelRow = $("<div></div>").attr({class: "row mt-3"});
    const labelCol = $("<div></div>")
        .attr({class: "col pt-2"})
        .appendTo(labelRow);
    const toggleButtonCol = $("<div></div>")
        .attr({class: "col text-end pb-2"})
        .appendTo(labelRow);

    labelCol.append(questionLabel);

    $("<button></button>")
        .attr({
            class: "btn btn-sm btn-danger",
            type: "button"
        })
        .click(e => toggleForm(e, form))
        .html("✕")
        .val("remove")
        .appendTo(toggleButtonCol);

    questionGroup.append(labelRow, questionInput);

    const answer = createAnswer(mcq.answer, i, j, 0, "success");
    const distractors = mcq.distractors.map((distractor, k) =>
        createAnswer(distractor, i, j, k + 1, "danger"));

    form.append(questionGroup, answer, ...distractors);

    return form;
};

const createAnswer = (answer, i, j, k, type) => {
    const answerId = `answer-${i}-${j}-${k}`;

    const border = type ? `border-${type}` : "";
    const bg = type ? `bg-${type}` : "";

    const answerGroup = $("<div></div>")
        .attr({class: "input-group mb-3"});

    const answerLabel = $("<div></div>")
        .attr({
            class: "input-group-prepend",
            style: "min-width: 7.5%"
        })
        .append(
            $("<span></span>")
                .text(LETTERS[k])
                .attr({
                    class: `input-group-text text-white font-weight-bold ${bg} w-100`
                })
        );
    const answerInput = $("<input>")
        .val(answer)
        .attr({
            id: answerId,
            type: "text",
            class: `form-control ${border}`,
            name: `answers`
        });

    answerGroup.append(answerLabel, answerInput);

    return answerGroup;
};

const toggleForm = (e, form) => {
    const button = $(e.target);
    const lineThrough = "text-decoration-line-through";
    if (button.val() === "remove") {
        $(form).find(`input[type=text]`)
            .prop("disabled", true)
            .addClass(lineThrough);
        button.removeClass("btn-danger")
            .addClass("btn-success")
            .html("✓")
            .val("restore");
    } else {
        button.removeClass("btn-success")
            .addClass("btn-danger")
            .html("✕")
            .val("remove");
        $(form).find(`input[type=text]`)
            .prop("disabled", false)
            .removeClass(lineThrough);
    }
};