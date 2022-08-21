const downloadQuestions = async () => {
    const previousLink = $("#getFileLink");
    if (previousLink.length) {
        previousLink.remove();
        // Adding some timeout to fake we are doing something
        await new Promise(r => setTimeout(r, 100));
    }

    const forms = $("[id^='mcq-']")
        .filter((_, form) => $(form).find("input[type=text]:disabled").length === 0);

    const texts = forms.map((i, form) => {
        const question = `${i + 1}. ${form.question.value}`;
        const answers = Array.from(form.answers).map(answer => answer.value);
        shuffle(answers);

        const answerStrings = answers.map((answer, j) =>
            LETTERS[j] + ") " + answer
        );

        return `
            <span style="font-weight: bold">${question}</span><br>${answerStrings.join("<br>")}
        `.trim();
    });

    const content = "<html><head><title>Generated MCQs</title></head><body>" +
        texts.map((_, text) =>
            `<div style="margin-bottom: 1em">${text}</div>`.trim()
        ).toArray().join("\n") + "</body></html>";

    download("mcqs-" + Date.now().toFixed().toString() + ".html", content);
};

const download = (filename, text) => {
    const existingButton = $("#getFileLink");
    const button = (
        existingButton.length ? existingButton :
        $('<a></a>')
            .text("Download")
            .attr({
                id: "getFileLink",
                target: "_blank",
                role: "button",
                class: "btn btn-success ms-3"
            })
    ).attr({
        href: "data:text/html;charset=utf-8," + encodeURIComponent(text),
        download: filename,
    });

    if (!existingButton.length) {
        $("#downloadBtnsCol").append(button);
    }
};

const shuffle = array => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
};
